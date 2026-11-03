"""ResNet-20 inference costed on DARTH-PUM.

[DARTH] Sec. 5.1 is pretty direct about this: "We place the layers of the
neural network (matrices) into analog arrays, while reserving digital
pipelines for input vectors." The whole network only needs ~784 of the chip's
~1844 tiles (see mapping.py) and stays resident the whole time, so one
inference instance sits in a small fixed corner of the chip and throughput
comes from just running a lot of instances in parallel elsewhere.

Split within one instance: conv + FC -> ACE (that's the analog MVM part),
everything else -> DCE (batch norm, ReLU, pooling, residual adds -- none of
that is a matrix-vector multiply, so it has no business being analog).

Weights and activations are both INT8. Every layer needs input bit-slicing (8
cycles, one bit at a time) and the weight matrix is bit-sliced across
cell_bits-wide arrays (mapping.py already bakes this into the tile count).
Layers with row-tiles (k*k*C_in > 64) also need a digital accumulate at the
end to sum up the R partial sums from R row-tiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from darthpum.ace import AnalogComputeElement
from darthpum.device import ACEParams, ADCParams
from darthpum.hct import HybridComputeTile

from .mapping import LayerMapping, map_resnet20

INPUT_BITS = 8          # INT8 activations
OUTPUT_BITS = 8         # INT8 accumulator output, post requantisation
VALUE_BITS = 8          # INT8 weights
CELL_BITS = 1           # SLC, matching [DARTH] Section 7.5's accuracy result


# [DARTH] Table 2: one ACE = 64 arrays. LayerMapping.tiles gives an ARRAY
# count (from ace.py's tiles_for_matrix), not an HCT count, so dividing by
# this converts "how many arrays does the resident network occupy" into "how
# many tiles does that take". Don't confuse this with the "waves" concept
# inside mvm_cost -- that's a TIMING question (how many sequential conversion
# rounds one MVM call needs if it alone exceeds 64 arrays), this is a
# CAPACITY question (how much area the whole, statically-placed network
# occupies). Related, not the same thing.
ARRAYS_PER_ACE = 64


@dataclass(frozen=True)
class LayerCost:
    name: str
    cycles: int
    energy_pJ: float
    tiles: int  # physical ARRAYS, not HCTs -- see ARRAYS_PER_ACE


def layer_cost(m: LayerMapping, tile: HybridComputeTile,
               reduction: str = "tree", active_row_fraction: float = 1.0) -> LayerCost:
    """Cost of one conv/fc layer, run once per output spatial position.

    We hand the full, unsliced weight_rows/weight_cols to
    AnalogComputeElement.mvm_cost, which works out the array footprint
    (row-tiles x col-tiles x bit-slices) and the wave count if that's more
    than one ACE's 64 arrays.

    What mvm_cost does NOT charge for: recombining row-tile partial sums
    (different row-tiles are different slices of the reduction dimension, so
    they need adding together), or the bit-sliced weight partial products
    (same shift-and-add idea the HCT does for input bit-slicing, see hct.py).
    Column-tiles are fine on their own -- different column-tiles are just
    different output channels, nothing to combine. So that missing digital
    accumulate is added right here.

    reduction controls how the partial sums get combined:
      "tree"   (default) pairwise tree, ceil(log2(n)) levels, each level's
               adds independent of each other and pipelineable in principle.
               Basically what any sane compiler/hardware would actually do.
      "serial" naive chain, n-1 dependent adds. Clearly worse, kept around
               only as a sensitivity check.

    active_row_fraction is the zero-skipping knob (sparsity.py), passed
    straight through to tile.mvm. 1.0 default = no change from before.
    """
    row_tiles = -(-m.weight_rows // 64)
    bit_slices = -(-VALUE_BITS // CELL_BITS)

    mvm = tile.mvm(
        rows=m.weight_rows, cols=m.weight_cols,
        input_bits=INPUT_BITS, output_bits=OUTPUT_BITS, value_bits=VALUE_BITS,
        active_row_fraction=active_row_fraction,
    )

    positions = m.out_shape[-1] * m.out_shape[-2] if m.kind == "conv" else 1
    cols_per_tile = min(m.weight_cols, 64)
    partials_to_combine = row_tiles * bit_slices

    accumulate_cycles = accumulate_energy = 0.0
    if partials_to_combine > 1:
        add = tile.dce.op_cost("ADD", width_bits=OUTPUT_BITS, elements=cols_per_tile)
        n_adds = (math.ceil(math.log2(partials_to_combine)) if reduction == "tree"
                 else partials_to_combine - 1)
        accumulate_cycles = n_adds * add.latency_cycles
        # energy is charged per pairwise add no matter the reduction shape --
        # same total work either way, it's only the LATENCY that depends on depth
        accumulate_energy = (partials_to_combine - 1) * add.energy_pJ

    return LayerCost(
        name=m.name,
        cycles=int((mvm.cycles + accumulate_cycles) * positions),
        energy_pJ=(mvm.energy_pJ + accumulate_energy) * positions,
        tiles=m.tiles,
    )


# non-MVM kernels: batch norm, ReLU, residual add, pooling -- all DCE

def nonmvm_cost(tile: HybridComputeTile, elements: int, has_bn: bool,
                has_relu: bool, has_residual: bool) -> LayerCost:
    """Per-position cost of the digital ops that follow a conv layer. BN folds
    to one multiply-add per channel (we model it as ADD -- DCE's cheapest
    multiply primitive is MUL8 and a scale-and-shift BN fold is the usual
    inference-time trick anyway); ReLU is the recipe book's RELUv1; a residual
    connection is just one more ADD."""
    cycles = energy = 0.0
    if has_bn:
        c = tile.digital_op("ADD", width_bits=OUTPUT_BITS, elements=elements)
        cycles += c.cycles; energy += c.energy_pJ
    if has_relu:
        c = tile.digital_op("RELUv1", width_bits=OUTPUT_BITS, elements=elements)
        cycles += c.cycles; energy += c.energy_pJ
    if has_residual:
        c = tile.digital_op("ADD", width_bits=OUTPUT_BITS, elements=elements)
        cycles += c.cycles; energy += c.energy_pJ
    return LayerCost("nonmvm", int(cycles), energy, tiles=0)


def inference_cost(
    tile: HybridComputeTile | None = None,
    ace_params: ACEParams | None = None,
    reduction: str = "tree",
    zero_skip: bool = False,
) -> tuple[list[LayerCost], list[LayerCost]]:
    """Full-network cost: (per-layer MVM costs, per-layer non-MVM costs).

    Every conv layer but the last gets BN+ReLU after it; the second conv in
    each BasicBlock also gets a residual add before its ReLU (standard
    post-activation ResNet, which is what this network actually is).

    zero_skip=True disables each layer's wordline drivers in proportion to
    its MEASURED activation sparsity (sparsity.py -- real zeros counted on
    real CIFAR-10 through our checkpoint, not a guessed rate), which cuts
    analog energy without touching cycles or accuracy. Default False
    reproduces every number from before this contribution was added.
    """
    tile = tile or HybridComputeTile(ace_params)
    mappings = map_resnet20(value_bits=VALUE_BITS, ace=tile.ace)

    skip_fractions = {}
    if zero_skip:
        from .sparsity import layer_skip_fractions
        skip_fractions = layer_skip_fractions()

    mvm_costs, nonmvm_costs = [], []
    for i, m in enumerate(mappings):
        active_fraction = 1.0 - skip_fractions.get(m.name, 0.0)
        mvm_costs.append(layer_cost(m, tile, reduction=reduction,
                                    active_row_fraction=active_fraction))

        if m.kind == "fc":
            continue  # no BN/ReLU after the final classifier

        elements = m.out_shape[0]  # per-channel BN/residual; ReLU per-element too
        is_second_conv_in_block = m.name.endswith(".conv2")
        nonmvm_costs.append(nonmvm_cost(
            tile, elements, has_bn=True, has_relu=True,
            has_residual=is_second_conv_in_block,
        ))

    return mvm_costs, nonmvm_costs


def hcts_per_instance(mvm_costs: list[LayerCost]) -> int:
    """Total array footprint -> HCTs needed to hold the network resident.
    (see ARRAYS_PER_ACE -- it's not just a straight sum)"""
    total_arrays = sum(c.tiles for c in mvm_costs)
    return -(-total_arrays // ARRAYS_PER_ACE)


if __name__ == "__main__":
    from darthpum.chip import Chip, Workload

    tile = HybridComputeTile()
    mvm_costs, nonmvm_costs = inference_cost(tile)

    total_mvm_cycles = sum(c.cycles for c in mvm_costs)
    total_nonmvm_cycles = sum(c.cycles for c in nonmvm_costs)
    total_cycles = total_mvm_cycles + total_nonmvm_cycles
    total_energy = sum(c.energy_pJ for c in mvm_costs + nonmvm_costs)
    total_arrays = sum(c.tiles for c in mvm_costs)
    total_tiles = hcts_per_instance(mvm_costs)

    print(f"{'layer':<12}{'cycles':>10}{'% of total':>12}")
    for c in sorted(mvm_costs, key=lambda c: -c.cycles)[:5]:
        print(f"{c.name:<12}{c.cycles:>10}{100*c.cycles/total_cycles:>11.1f}%")
    print("...")

    print(f"\nMVM (ACE) cycles    : {total_mvm_cycles:>10,}  "
          f"({100*total_mvm_cycles/total_cycles:.1f}%)")
    print(f"non-MVM (DCE) cycles: {total_nonmvm_cycles:>10,}  "
          f"({100*total_nonmvm_cycles/total_cycles:.1f}%)")
    print(f"TOTAL per inference : {total_cycles:>10,} cycles "
          f"({total_cycles/1000:.2f} us at 1 GHz)")
    print(f"array footprint     : {total_arrays} physical arrays")
    print(f"HCTs per instance   : {total_tiles} (ceil({total_arrays}/{ARRAYS_PER_ACE}), of ~1844 on the chip)")

    w = Workload(
        name="ResNet-20", tiles_per_instance=total_tiles,
        cycles_per_instance=total_cycles, energy_pJ_per_instance=total_energy,
        unit="inferences",
    )
    r = Chip().run(w)
    print(f"\nchip-wide: {r.instances_parallel} parallel instances, "
          f"{r.throughput_per_s:.1f} inferences/s, {r.power_W:.1f} W"
          + ("  [thermally limited]" if r.thermally_limited else ""))
