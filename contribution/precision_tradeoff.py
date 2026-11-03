"""How many bits per analog cell should ResNet-20 actually use?

The paper only runs ResNet-20 in SLC mode (1 bit/cell). Push more bits into
each cell and you need fewer physical arrays for the same matrix -- fewer
bit-slices -- so more copies of the network fit on the chip and throughput
climbs. Tradeoff: a cell with more distinguishable levels is harder to
program accurately, so accuracy drops. Wanted to actually measure this
instead of just assuming SLC was the right call.

Throughput side was easy -- ACEParams already has cell_bits and
tiles_for_matrix divides by it, so this file just runs the sweep over the
three settings.

Accuracy side reuses the exact CrossSim pipeline from the paper's own SLC
number (see workloads/resnet20/accuracy.py), just sweeping weight_bits /
Nslices so cell_bits lines up with the throughput side: Nslices 8/4/2 ->
cell_bits 1/2/4 (weight_bits pinned at 8 throughout).

Numbers, full 10k CIFAR-10 test set, our checkpoint:

    cell_bits   tiles/instance   chip throughput     accuracy
    1 (SLC)     13               10364.7 inf/s       83.43%
    2            7               19453.5 inf/s (1.88x)   70.68%  (-12.8pp)
    4            4               34099.1 inf/s (3.29x)   53.84%  (-29.6pp)

Unlike zero-skipping, this one doesn't get diluted -- every layer shrinks by
the same cell_bits factor so the gain shows up almost untouched at the chip
level. The honest catch is on the accuracy side: the drop is real and not
small, so SLC isn't some obviously-left-on-the-table mistake, it's a
reasonable spot on a curve that gets worse fast in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass

CELL_BITS_SWEPT = (1, 2, 4)

# weight_bits stays at 8 (INT8, same as everywhere else here); Nslices just
# chops that into cell_bits-wide slices.
NSLICES_FOR_CELL_BITS = {1: 8, 2: 4, 4: 2}

# full 10k CIFAR-10 test set, RRAMMilo noise, SAR ADC -- same config as the
# paper's SLC number. regenerate with measure_accuracy() if these ever need
# re-checking (takes a while, see the docstring there)
MEASURED_ACCURACY_PCT = {1: 83.43, 2: 70.68, 4: 53.84}


@dataclass(frozen=True)
class PrecisionPoint:
    cell_bits: int
    tiles_per_instance: int
    chip_throughput_per_s: float
    accuracy_pct: float


def throughput_sweep() -> list[PrecisionPoint]:
    """Runs the chip model once per cell_bits setting and collects tiles/
    instance + throughput for each."""
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
    # redoes one entry of MEASURED_ACCURACY_PCT from scratch -- needs torch,
    # CrossSim and the checkpoint on disk. full 10k run takes a few minutes,
    # don't call this in a loop
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
