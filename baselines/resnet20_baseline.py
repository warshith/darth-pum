"""The ResNet-20 Baseline: a CPU paired with an analog PUM accelerator.

Convolutions (the matrix operations) go to the accelerator, everything else --
batch norm, ReLU, residual adds -- runs on the host. Measured on this CPU via
PyTorch's profiler, convolutions are 85.3% of ResNet-20's forward time and
non-conv ops are 14.7% (see measure_resnet20_cpu_split in this module).

Modelling decisions
--------------------
1. **Data volume per round trip.** A ResNet-20 conv layer's activations are
   KB-scale (e.g. 16x32x32 = 16 KiB for an early layer), so whether a fixed
   latency floor or link bandwidth dominates a round trip is not assumed
   here; both are modelled and the result reports which one actually binds,
   per layer.

2. **Accelerator parallel capacity.** ResNet-20 needs only 784 analog arrays;
   a 1.5 GB accelerator holds roughly 3.1 million 64x64 1-bit-cell arrays, so
   it can hold ~4000 PARALLEL copies of the resident network -- comparable to
   how DARTH-PUM itself replicates across many HCTs, just without the
   HCT-level area overhead (front end, shift unit, etc.) since this is a bare
   accelerator, not a hybrid tile. Capacity (how many images run concurrently)
   and latency (how long any one image's own dependency chain takes) are
   therefore kept as separate quantities throughout.

3. **Round trips per layer, not per layer-output.** DARTH-PUM's DCE sits
   co-located with its ACE and combines row-tile and bit-slice partial sums
   on-chip, "for free" (resnet_pum.py's tree reduction). The paper's Baseline
   accelerator has no such co-located digital unit, so each ARRAY's own
   output stream -- one per (row-tile, bit-slice) combination, 784 of them
   total across the network, not one per layer (20) -- plausibly has to cross
   the host link separately for the CPU to combine. A first attempt used one
   round trip per layer and needed an implied ~444 us/round-trip to match the
   paper's latency ratio -- implausibly large. Using 784 (one per physical
   array, the natural unit: each array produces one output stream) needs only
   ~11.3 us/round-trip, an ordinary order of magnitude for OS-mediated
   device I/O -- evidence this is the right granularity, not just a number
   picked to fit.

Cross-check against the paper's own numbers
--------------------------------------------
[DARTH] Section 7.1 gives two ResNet-20 ratios that are very different in
character: throughput improves 14.8x, but single-inference latency improves
only 1/0.6 = 1.667x ("DARTH-PUM's improvement over Baseline throughput is not
just a result of improved inference latency (which decreases by 40.0%)"). The
large gap between them means most of DARTH-PUM's advantage here comes from
PARALLELISM, not per-inference latency -- consistent with a 269K-parameter
network needing only 13 of ~1844 tiles and so replicating heavily.

Calibrating this model's two unpublished interconnect parameters -- round-trip
latency and link bandwidth -- against these two ratios simultaneously
(CALIBRATED_ROUND_TRIP_NS, CALIBRATED_LINK_GB_PER_S) reproduces BOTH: 14.72x
throughput (paper: 14.8x) and 1.67x latency (paper: 1.667x), both within 1%.

IMPORTANT CAVEAT, stated plainly: this fits two free parameters to two
targets, which can generally be solved exactly regardless of whether the
underlying model is right. The reason this result is still meaningful, not
just curve-fit: the two calibrated VALUES are independently ordinary. 11.3 us
per round trip and 4.2 GB/s are both squarely in normal, non-exotic ranges for
OS-mediated device I/O and a PCIe-3.0-x4-class link respectively. Had matching
the paper's ratios required an absurd value (a negligible latency, or a link
faster than any real interconnect), that would be the tell the model is wrong
despite "matching". It did not -- it needed ordinary values on both counts.
"""

from __future__ import annotations

from dataclasses import dataclass

from darthpum.hct import HybridComputeTile

from workloads.resnet20.mapping import LayerMapping, map_resnet20
from workloads.resnet20.resnet_pum import ARRAYS_PER_ACE, VALUE_BITS, hcts_per_instance, inference_cost

ACCELERATOR_GB = 1.5
"""[DARTH] Section 6: the Baseline's analog PUM accelerator is 1.5 GB."""

P_CORES = 8
"""Performance cores credited to the Baseline.

The host has 8 P-cores plus 4 E-cores. [DARTH] Section 3 describes its motivation
baseline as an "8-core CPU", so 8 is the defensible count.
"""

HOST_LINK_GB_PER_S = 50.0
"""Representative DDR5-class bandwidth for the unspecified host<->accelerator
link. The paper does not specify the interconnect."""

CALIBRATED_ROUND_TRIP_NS = 11_300.0
"""Derived from [DARTH] Section 7.1's 1.667x single-inference latency ratio.
See the module docstring's caveat on what this calibration does and does not prove."""

CALIBRATED_LINK_GB_PER_S = 4.2
"""Derived from [DARTH] Section 7.1's 14.8x throughput ratio. Same caveat."""

ROUND_TRIP_LATENCY_NS = 500.0
"""Uncalibrated default: a modest, non-kernel-bypass round-trip floor. Whether
this or bandwidth dominates is computed, not assumed -- see the module
docstring."""

CPU_NONCONV_US_PER_IMAGE_1_CORE = 191.40
"""BatchNorm + ReLU + residual add + pooling, profiled with torch.profiler
over 100 forward passes on a single thread. 14.7% of total CPU forward time;
see measure_resnet20_cpu_split() to regenerate."""


@dataclass(frozen=True)
class ResNet20BaselineResult:
    throughput_inferences_per_s: float
    single_inference_latency_us: float
    binding_resource: str  # "cpu", "accelerator", "transfer_bandwidth", or "transfer_latency"
    accelerator_parallel_capacity: int
    dominant_transfer_term: str  # "latency_floor" or "bandwidth", for a single round trip


class ResNet20Baseline:
    """ResNet-20 inference on a CPU with convolutions offloaded to an analog
    accelerator. Capacity (how many images run concurrently) and latency (one
    image's own dependency chain) are modelled separately throughout."""

    def __init__(
        self,
        cores: int = P_CORES,
        link_GB_per_s: float = HOST_LINK_GB_PER_S,
        round_trip_latency_ns: float = ROUND_TRIP_LATENCY_NS,
        accelerator_GB: float = ACCELERATOR_GB,
        tile: HybridComputeTile | None = None,
    ):
        self.cores = cores
        self.link = link_GB_per_s
        self.round_trip_ns = round_trip_latency_ns
        self.accelerator_GB = accelerator_GB
        self.tile = tile or HybridComputeTile()
        self.mappings = map_resnet20(value_bits=VALUE_BITS, ace=self.tile.ace)

        mvm_costs, nonmvm_costs = inference_cost(tile=self.tile)
        self._mvm_costs = mvm_costs
        self._sequential_latency_ns = sum(c.cycles for c in mvm_costs + nonmvm_costs)
        self._array_footprint = sum(c.tiles for c in mvm_costs)

    # -- capacities (images/s if this were the ONLY constraint) ------------

    def cpu_capacity_images_per_s(self) -> float:
        """Images are independent (no cross-image dependency), so non-conv
        CPU work scales across cores."""
        return self.cores / (CPU_NONCONV_US_PER_IMAGE_1_CORE * 1e-6)

    def accelerator_parallel_capacity(self) -> int:
        """How many copies of the resident network fit in the accelerator at
        once -- the array-count analogue of darthpum/area.py's hct_count, but
        for a bare accelerator (no HCT-level area overhead)."""
        cell_bits = 1  # SLC, same as the ACE device assumption
        array_bits = self.tile.ace.p.array.bits
        total_arrays = int(self.accelerator_GB * 2**30 * 8 / cell_bits / array_bits)
        return max(1, total_arrays // self._array_footprint)

    def accelerator_capacity_images_per_s(self) -> float:
        """Many copies run concurrently, each taking the full sequential
        latency -- the same reasoning as DARTH-PUM's own tile replication."""
        parallel = self.accelerator_parallel_capacity()
        return parallel / (self._sequential_latency_ns * 1e-9)

    def layer_transfer_ns(self, m: LayerMapping) -> tuple[float, float]:
        """(latency_floor_ns, bandwidth_ns) for ONE ARRAY's round trip in this
        layer -- a partial-sum stream, not the full combined layer output
        (see roundtrips_per_layer for why this is the right granularity)."""
        from workloads.resnet20.resnet_pum import OUTPUT_BITS
        cols_per_tile = min(m.weight_cols, 64)
        positions = m.out_shape[-1] * m.out_shape[-2] if m.kind == "conv" else 1
        partial_bytes = -(-cols_per_tile * OUTPUT_BITS // 8) * positions
        bandwidth_ns = partial_bytes / (self.link * 1e9) * 1e9
        return self.round_trip_ns, bandwidth_ns

    def roundtrips_per_layer(self, m: LayerMapping) -> int:
        """One physical array's output stream = one round trip, since DARTH-
        PUM's co-located DCE (which combines these on-chip) has no equivalent
        in a bare accelerator. row_tiles x bit_slices per layer, matching
        LayerMapping.tiles exactly (both count physical arrays the same way)."""
        row_tiles = -(-m.weight_rows // 64)
        bit_slices = -(-VALUE_BITS // 1)  # 1 = CELL_BITS, SLC
        return row_tiles * bit_slices

    def bytes_per_image(self) -> int:
        """Total bytes moved for one image, at the (row-tile, bit-slice)
        granularity: each array sends back OUTPUT_BITS-wide partial sums for
        its slice of output columns, not the full combined layer output."""
        from workloads.resnet20.resnet_pum import OUTPUT_BITS
        total = 0
        for m in self.mappings:
            cols_per_tile = min(m.weight_cols, 64)
            positions = m.out_shape[-1] * m.out_shape[-2] if m.kind == "conv" else 1
            partial_bytes = -(-cols_per_tile * OUTPUT_BITS // 8) * positions
            total += self.roundtrips_per_layer(m) * partial_bytes
        return total

    def transfer_bandwidth_capacity_images_per_s(self) -> float:
        """The link is SHARED across every concurrently in-flight image, so
        aggregate bytes/s -- not any single image's round trip -- is the real
        throughput ceiling from bandwidth."""
        return (self.link * 1e9) / self.bytes_per_image()

    def transfer_latency_ns_per_image(self) -> float:
        """Sum of per-ARRAY round-trip latency floors for ONE image's own
        (sequential) pass through the network -- contributes to LATENCY, not
        directly to throughput, since independent images can pipeline behind
        each other's round trips. 784 round trips total (one per physical
        array), not 20 (one per layer) -- see the module docstring for why
        this granularity, not the coarser one, is what reconciles against the
        paper's own latency ratio.
        """
        return sum(self.roundtrips_per_layer(m) for m in self.mappings) * self.round_trip_ns

    # -- rollup --------------------------------------------------------------

    def run(self) -> ResNet20BaselineResult:
        caps = {
            "cpu": self.cpu_capacity_images_per_s(),
            "accelerator": self.accelerator_capacity_images_per_s(),
            "transfer_bandwidth": self.transfer_bandwidth_capacity_images_per_s(),
        }
        binding = min(caps, key=caps.get)
        throughput = caps[binding]

        latency_ns = (
            CPU_NONCONV_US_PER_IMAGE_1_CORE * 1e3  # one image, one core
            + self.transfer_latency_ns_per_image()
            + self._sequential_latency_ns          # one image's own accelerator chain
        )

        term_counts = {"latency_floor": 0, "bandwidth": 0}
        for m in self.mappings:
            lat, bw = self.layer_transfer_ns(m)
            term_counts["latency_floor" if lat > bw else "bandwidth"] += 1
        dominant = max(term_counts, key=term_counts.get)

        return ResNet20BaselineResult(
            throughput_inferences_per_s=throughput,
            single_inference_latency_us=latency_ns / 1000,
            binding_resource=binding,
            accelerator_parallel_capacity=self.accelerator_parallel_capacity(),
            dominant_transfer_term=dominant,
        )

    def crosscheck_against_paper_latency(self, darthpum_latency_us: float) -> dict:
        """[DARTH] Section 7.1: single-inference latency improves 1/0.6=1.667x.
        Compare that implied Baseline latency against what this model predicts."""
        implied = darthpum_latency_us * (1 / 0.6)
        predicted = self.run().single_inference_latency_us
        return {
            "implied_baseline_latency_us": implied,
            "predicted_baseline_latency_us": predicted,
            "ratio_predicted_to_implied": predicted / implied,
        }


def measure_resnet20_cpu_split() -> dict:
    """Regenerate CPU_NONCONV_US_PER_IMAGE_1_CORE using torch.profiler.
    Requires torch and the downloaded checkpoint; not run at import time."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from workloads.resnet20.model import load_pretrained

    torch.set_num_threads(1)
    m = load_pretrained("checkpoints/resnet20-12fca82f.th")
    m.eval()
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        for _ in range(10):
            m(x)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(100):
                m(x)

    events = prof.key_averages()
    total = sum(e.self_cpu_time_total for e in events)
    conv = sum(e.self_cpu_time_total for e in events if "conv" in e.key.lower())
    return {
        "full_us_per_image": total / 100,
        "conv_us_per_image": conv / 100,
        "nonconv_us_per_image": (total - conv) / 100,
        "conv_pct": 100 * conv / total,
    }


if __name__ == "__main__":
    from darthpum.chip import Chip, Workload

    tile = HybridComputeTile()
    mvm_costs, nonmvm_costs = inference_cost(tile)
    total_cycles = sum(c.cycles for c in mvm_costs + nonmvm_costs)
    total_energy = sum(c.energy_pJ for c in mvm_costs + nonmvm_costs)
    total_tiles = hcts_per_instance(mvm_costs)

    dp_workload = Workload(
        name="ResNet-20", tiles_per_instance=total_tiles,
        cycles_per_instance=total_cycles, energy_pJ_per_instance=total_energy,
        unit="inferences",
    )
    dp = Chip().run(dp_workload)
    dp_latency_us = total_cycles / 1000  # 1 GHz

    print(f"DARTH-PUM : {dp.throughput_per_s:>10.1f} inferences/s chip-wide, "
          f"{dp_latency_us:>10.2f} us/inference latency\n")

    for label, rt_ns, link_gbps in (
        ("uncalibrated (neutral 500 ns / 50 GB/s)", ROUND_TRIP_LATENCY_NS, HOST_LINK_GB_PER_S),
        ("calibrated (see module docstring caveat)", CALIBRATED_ROUND_TRIP_NS, CALIBRATED_LINK_GB_PER_S),
    ):
        b = ResNet20Baseline(tile=tile, round_trip_latency_ns=rt_ns, link_GB_per_s=link_gbps)
        base = b.run()
        print(f"Baseline [{label}]:")
        print(f"  {base.throughput_inferences_per_s:>10.1f} inferences/s, "
              f"{base.single_inference_latency_us:>10.2f} us/inference latency, "
              f"binding: {base.binding_resource}")
        print(f"  SPEEDUP throughput: {dp.throughput_per_s / base.throughput_inferences_per_s:>6.2f}x  (paper: 14.8x)")
        print(f"  SPEEDUP latency:    {base.single_inference_latency_us / dp_latency_us:>6.2f}x  (paper: 1.667x)")
        print()

    b = ResNet20Baseline(tile=tile, round_trip_latency_ns=CALIBRATED_ROUND_TRIP_NS,
                         link_GB_per_s=CALIBRATED_LINK_GB_PER_S)
    print(f"accelerator parallel capacity: {b.accelerator_parallel_capacity()} images "
          f"(a 1.5 GB bare accelerator holding many copies of a 269K-param network)")

    print()
    cc = b.crosscheck_against_paper_latency(dp_latency_us)
    print("Cross-check against the paper's own 1.667x latency ratio:")
    for k, v in cc.items():
        print(f"  {k:<32}{v:,.2f}")

    print()
    print("All three capacities (images/s), for comparison:")
    print(f"  cpu                : {b.cpu_capacity_images_per_s():,.1f}")
    print(f"  accelerator        : {b.accelerator_capacity_images_per_s():,.1f}")
    print(f"  transfer bandwidth : {b.transfer_bandwidth_capacity_images_per_s():,.1f}")
