"""ResNet-20 on a CPU + analog PUM accelerator -- the paper's "Baseline".

Convs go to the accelerator, everything else (BN, ReLU, residual adds) runs on
the host CPU. Measured convs are ~85% of ResNet-20's forward time on this
machine (see measure_resnet20_cpu_split below if you want to regenerate that).

A few modelling calls worth flagging up front:

- Activations here are tiny (16x32x32 = 16 KiB for an early layer), so we
  don't just assume a latency floor or bandwidth dominates each round trip --
  both are modelled and we check which one actually wins, per layer.
- The accelerator is way bigger than the network needs. ResNet-20 only uses
  784 arrays out of the ~3.1M that fit in a 1.5 GB chip, so it ends up
  running ~4000 copies of the network side by side. Capacity (how many
  images at once) and latency (one image's own critical path) get tracked as
  two separate numbers the whole way through -- conflating them was an early
  mistake that made the Baseline look absurdly slow.
- Round trips are counted per physical array (784 total), not per layer (20).
  DARTH-PUM's DCE sits right next to its ACE and combines partial sums
  on-chip for free; a bare accelerator has nothing playing that role, so each
  array's output plausibly has to cross the host link on its own. Doing this
  per-layer first needed an implied ~444 us/round-trip to match the paper's
  latency ratio, which is a ridiculous number for device I/O. Per-array needs
  only ~11.3 us, which is normal.

[DARTH] Section 7.1 gives two very different ResNet-20 ratios: 14.8x
throughput but only 1.667x single-inference latency. So most of the speedup
is coming from parallelism, not per-inference speed -- which tracks, since a
269K-parameter network only needs 13 of ~1844 tiles and gets replicated a lot.

The two unpublished interconnect numbers (round-trip latency, link
bandwidth) are calibrated to hit both of those ratios at once, see the
CALIBRATED_* constants below. That lands at 14.72x / 1.67x, both within 1% of
the paper. Fitting two free parameters to two targets will basically always
"work" on its own -- what makes this worth something is that the two
calibrated values land on ordinary numbers (11.3 us round trip, 4.2 GB/s
link) instead of something absurd that only matches because it was forced to.
"""

from __future__ import annotations

from dataclasses import dataclass

from darthpum.hct import HybridComputeTile

from workloads.resnet20.mapping import LayerMapping, map_resnet20
from workloads.resnet20.resnet_pum import ARRAYS_PER_ACE, VALUE_BITS, hcts_per_instance, inference_cost

ACCELERATOR_GB = 1.5  # [DARTH] Sec 6: the Baseline's analog PUM accelerator is 1.5 GB

# host is 8 P-cores + 4 E-cores; the paper just says "8-core CPU" for this
# baseline so we credit it 8, not 12.
P_CORES = 8

HOST_LINK_GB_PER_S = 50.0  # rough DDR5-class number; paper doesn't specify an interconnect

# backed out from [DARTH] 7.1's 1.667x latency ratio -- see module docstring for the caveat
CALIBRATED_ROUND_TRIP_NS = 11_300.0

# backed out from [DARTH] 7.1's 14.8x throughput ratio -- same caveat applies
CALIBRATED_LINK_GB_PER_S = 4.2

ROUND_TRIP_LATENCY_NS = 500.0  # plain uncalibrated default, not tuned to match anything

CPU_NONCONV_US_PER_IMAGE_1_CORE = 191.40
# BatchNorm + ReLU + residual add + pooling, profiled with torch.profiler over
# 100 forward passes on one thread -- 14.7% of total forward time. Rerun
# measure_resnet20_cpu_split() below to regenerate this if the checkpoint changes.


@dataclass(frozen=True)
class ResNet20BaselineResult:
    throughput_inferences_per_s: float
    single_inference_latency_us: float
    binding_resource: str  # "cpu", "accelerator", "transfer_bandwidth", or "transfer_latency"
    accelerator_parallel_capacity: int
    dominant_transfer_term: str  # "latency_floor" or "bandwidth", for a single round trip


class ResNet20Baseline:
    """ResNet-20 on a CPU with convs offloaded to an analog accelerator.

    Capacity (images running at once) and latency (one image's own
    dependency chain) are kept as separate numbers throughout.
    """

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
        """Images don't depend on each other, so non-conv CPU work just scales across cores."""
        return self.cores / (CPU_NONCONV_US_PER_IMAGE_1_CORE * 1e-6)

    def accelerator_parallel_capacity(self) -> int:
        """How many copies of the network fit in the accelerator at once.

        Same idea as area.py's hct_count, just for a bare accelerator with no
        HCT-level overhead (no front end, no shift unit, none of that).
        """
        cell_bits = 1  # SLC, same assumption as the ACE
        array_bits = self.tile.ace.p.array.bits
        total_arrays = int(self.accelerator_GB * 2**30 * 8 / cell_bits / array_bits)
        return max(1, total_arrays // self._array_footprint)

    def accelerator_capacity_images_per_s(self) -> float:
        """All the copies run concurrently, each still taking the full sequential latency."""
        parallel = self.accelerator_parallel_capacity()
        return parallel / (self._sequential_latency_ns * 1e-9)

    def layer_transfer_ns(self, m: LayerMapping) -> tuple[float, float]:
        """(latency_floor_ns, bandwidth_ns) for one array's round trip in this layer.

        This is a partial-sum stream, not the combined layer output -- see
        roundtrips_per_layer for why that's the right unit here.
        """
        from workloads.resnet20.resnet_pum import OUTPUT_BITS
        cols_per_tile = min(m.weight_cols, 64)
        positions = m.out_shape[-1] * m.out_shape[-2] if m.kind == "conv" else 1
        partial_bytes = -(-cols_per_tile * OUTPUT_BITS // 8) * positions
        bandwidth_ns = partial_bytes / (self.link * 1e9) * 1e9
        return self.round_trip_ns, bandwidth_ns

    def roundtrips_per_layer(self, m: LayerMapping) -> int:
        """One physical array's output = one round trip (no co-located DCE here
        to combine them on-chip the way DARTH-PUM does). row_tiles x bit_slices,
        matching LayerMapping.tiles -- both count physical arrays the same way."""
        row_tiles = -(-m.weight_rows // 64)
        bit_slices = -(-VALUE_BITS // 1)  # 1 = CELL_BITS, SLC
        return row_tiles * bit_slices

    def bytes_per_image(self) -> int:
        """Total bytes moved for one image, at (row-tile, bit-slice) granularity:
        each array sends back its own OUTPUT_BITS-wide partial sums, not the
        full combined layer output."""
        from workloads.resnet20.resnet_pum import OUTPUT_BITS
        total = 0
        for m in self.mappings:
            cols_per_tile = min(m.weight_cols, 64)
            positions = m.out_shape[-1] * m.out_shape[-2] if m.kind == "conv" else 1
            partial_bytes = -(-cols_per_tile * OUTPUT_BITS // 8) * positions
            total += self.roundtrips_per_layer(m) * partial_bytes
        return total

    def transfer_bandwidth_capacity_images_per_s(self) -> float:
        """The link is shared across every in-flight image, so it's aggregate
        bytes/s -- not any single image's round trip -- that caps throughput."""
        return (self.link * 1e9) / self.bytes_per_image()

    def transfer_latency_ns_per_image(self) -> float:
        """Sum of per-array round-trip floors for one image's sequential pass
        through the network. 784 round trips, not 20 -- see the module
        docstring for why that granularity is the one that actually matches
        the paper's latency ratio."""
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
        """[DARTH] 7.1: single-inference latency improves 1/0.6 = 1.667x.
        Compare the implied Baseline latency against what this model predicts."""
        implied = darthpum_latency_us * (1 / 0.6)
        predicted = self.run().single_inference_latency_us
        return {
            "implied_baseline_latency_us": implied,
            "predicted_baseline_latency_us": predicted,
            "ratio_predicted_to_implied": predicted / implied,
        }


def measure_resnet20_cpu_split() -> dict:
    """Regenerate CPU_NONCONV_US_PER_IMAGE_1_CORE with torch.profiler.
    Needs torch + the downloaded checkpoint; not run at import time."""
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
