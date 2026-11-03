"""Contribution: how many bits per analog cell should ResNet-20 use?

The paper only evaluates ResNet-20 in SLC mode (one bit per cell). Storing
more bits per cell needs fewer physical arrays for the same matrix, since
fewer bit-slices are required -- which means more copies of the network fit
on the chip and throughput goes up. The cost is that a cell holding more
distinguishable levels is harder to program precisely, so accuracy goes down.
This module measures that tradeoff directly instead of assuming SLC is the
only reasonable choice.

Throughput side: darthpum.device.ACEParams already exposes cell_bits, and
AnalogComputeElement.tiles_for_matrix divides by it directly, so every piece
needed already exists and is tested -- this module only runs the sweep.

Accuracy side: measured with the same CrossSim pipeline used for the paper's
own SLC accuracy result (workloads/resnet20/accuracy.py), varying weight_bits
and Nslices so the resulting cell_bits matches the throughput side exactly
(weight_bits=8, Nslices=8/4/2 gives cell_bits=1/2/4).

Results, full 10,000-image CIFAR-10 test set, our verified checkpoint:

    cell_bits   tiles/instance   chip throughput     accuracy
    1 (SLC)     13               10364.7 inf/s       83.43%
    2            7               19453.5 inf/s (1.88x)   70.68%  (-12.8pp)
    4            4               34099.1 inf/s (3.29x)   53.84%  (-29.6pp)

Unlike the zero-skipping contribution, this effect is NOT diluted by other
kernels: every layer's tile footprint shrinks by the same factor, so the
throughput gain reaches the whole-network number intact. The honest finding
is the other direction -- the accuracy cost is real and sizeable, so SLC is
not an obviously suboptimal choice the paper left on the table; it's a
reasonable point on a tradeoff curve that gets considerably worse in both
directions past it.
"""

from __future__ import annotations

from dataclasses import dataclass

CELL_BITS_SWEPT = (1, 2, 4)

# weight_bits is fixed at 8 (INT8, matching every other result in this
# project); Nslices divides it into that many cell_bits-wide slices.
NSLICES_FOR_CELL_BITS = {1: 8, 2: 4, 4: 2}

MEASURED_ACCURACY_PCT = {1: 83.43, 2: 70.68, 4: 53.84}
"""Full 10,000-image CIFAR-10 test set, RRAMMilo noise, SAR ADC -- the same
configuration as the paper's own SLC accuracy result. Regenerate with
measure_accuracy()."""


@dataclass(frozen=True)
class PrecisionPoint:
    cell_bits: int
    tiles_per_instance: int
    chip_throughput_per_s: float
    accuracy_pct: float


def throughput_sweep() -> list[PrecisionPoint]:
    """Tiles/instance and chip-wide throughput at each cell_bits setting."""
    from darthpum.chip import Chip, Workload
    from darthpum.device import ACEParams
    from darthpum.hct import HybridComputeTile
    from workloads.resnet20.resnet_pum import hcts_per_instance, inference_cost

    out = []
    for cell_bits in CELL_BITS_SWEPT:
        ace_params = ACEParams(cell_bits=cell_bits)
        tile = HybridComputeTile(ace_params=ace_params)
        mvm_costs, nonmvm_costs = inference_cost(tile)
        total_cycles = sum(c.cycles for c in mvm_costs + nonmvm_costs)
        total_energy = sum(c.energy_pJ for c in mvm_costs + nonmvm_costs)
        tiles_needed = hcts_per_instance(mvm_costs)
        w = Workload("ResNet-20", tiles_needed, total_cycles, total_energy,
                    unit="inferences")
        r = Chip(ace_params=ace_params).run(w)
        out.append(PrecisionPoint(
            cell_bits=cell_bits, tiles_per_instance=tiles_needed,
            chip_throughput_per_s=r.throughput_per_s,
            accuracy_pct=MEASURED_ACCURACY_PCT[cell_bits],
        ))
    return out


def measure_accuracy(cell_bits: int, n_images: int = 10_000,
                     batch_size: int = 200) -> float:
    """Regenerate one entry of MEASURED_ACCURACY_PCT from scratch. Requires
    torch/CrossSim, the checkpoint, and the CIFAR-10 test set. Takes several
    minutes at n_images=10000."""
    from simulator.algorithms.dnn.torch.convert import convertible_modules, from_torch

    from workloads.resnet20.accuracy import CHECKPOINT, _device_params, evaluate
    from workloads.resnet20.model import load_pretrained

    nslices = NSLICES_FOR_CELL_BITS[cell_bits]
    ref = load_pretrained(CHECKPOINT)
    n_layers = len(convertible_modules(ref))

    model = load_pretrained(CHECKPOINT)
    params = [_device_params(weight_bits=8, Nslices=nslices, ideal=False)
             for _ in range(n_layers)]
    analog = from_torch(model, params, fuse_batchnorm=True, bias_rows=0)
    analog.eval()
    return evaluate(analog, n_images=n_images, batch_size=batch_size)


if __name__ == "__main__":
    points = throughput_sweep()
    base = points[0]
    print(f"{'cell_bits':<10}{'tiles/inst':>11}{'throughput':>14}{'speedup':>9}{'accuracy':>10}")
    for p in points:
        speedup = p.chip_throughput_per_s / base.chip_throughput_per_s
        print(f"{p.cell_bits:<10}{p.tiles_per_instance:>11}"
              f"{p.chip_throughput_per_s:>14,.1f}{speedup:>8.2f}x{p.accuracy_pct:>9.2f}%")
