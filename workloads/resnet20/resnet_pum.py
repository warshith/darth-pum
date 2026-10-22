"""ResNet-20 inference costed on DARTH-PUM.

[DARTH] Section 5.1: "We place the layers of the neural network (matrices) into
analog arrays, while reserving digital pipelines for input vectors." All layers'
weights stay resident simultaneously (the whole network needs only 784 of the
chip's ~1844 tiles -- see mapping.py), so one inference instance occupies a
small, fixed slice of the chip and throughput comes from running many instances
in parallel across the rest.

Within one instance, the paper is explicit about the split: convolution and FC
go to the ACE (analog MVM); everything else -- batch norm, ReLU, pooling,
residual adds -- goes to the DCE, since none of those are matrix-vector
multiplies.

Convolution weights and activations are INT8, so each layer needs input
bit-slicing (8 cycles, one input bit per cycle) and the matrix itself is
bit-sliced across cell_bits-wide arrays (mapping.py's physical tile count
already accounts for this). Row-tiles (when k*k*C_in > 64) also need a final
digital accumulate -- a layer with R row-tiles produces R partial sums that
must be summed.
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


ARRAYS_PER_ACE = 64
"""[DARTH] Table 2: one ACE holds 64 arrays. LayerMapping.tiles (ace.py's
tiles_for_matrix) returns an ARRAY count, not an HCT count, so the network's
total array footprint must be divided by ARRAYS_PER_ACE to get the number of
tiles needed to hold it resident. Layers are placed statically and held
resident simultaneously ([DARTH] Section 5.1), so this is an AREA/capacity
question (how many ACEs' worth of array footprint the resident network
occupies), separate from the WAVES concept inside
AnalogComputeElement.mvm_cost, which answers a TIMING question instead: if
one MVM call alone needs more arrays than one ACE has, how many sequential
conversion rounds are needed. Both matter, but they are not interchangeable.
"""


@dataclass(frozen=True)
class LayerCost:
    name: str
    cycles: int
    energy_pJ: float
    tiles: int
    """Physical ARRAYS this layer occupies (not HCTs -- see ARRAYS_PER_ACE)."""


def layer_cost(m: LayerMapping, tile: HybridComputeTile,
               reduction: str = "tree") -> LayerCost:
    """Cost of one conv/fc layer, run once per output spatial position.

    Pass the FULL (unsliced) weight_rows/weight_cols straight to
    AnalogComputeElement.mvm_cost -- it already computes the correct array
    footprint (row-tiles x col-tiles x bit-slices) and the resulting wave count
    if that exceeds one ACE's 64-array capacity. An earlier version pre-sliced
    to a single <=64-row tile and charged that cost once, silently DROPPING the
    other row-tiles' MVM passes entirely for any layer wider than 64 rows --
    caught by checking per-layer cycle counts against a rough hand calculation
    before trusting the full-network total.

    What mvm_cost does NOT do is charge for recombining the row-tile partial
    sums (different row-tiles cover different reduction-dimension slices of the
    SAME output, so their results must be added) and the bit-sliced weight
    partial products (different bit-slices must be shifted and added, the same
    shift-and-add pattern the HCT's coordination layer implements for input
    bit-slicing -- see hct.py). Column-tiles need no recombination: different
    column-tiles are different, independent output channels. That digital
    accumulation is added here explicitly.

    `reduction` picks how the partial sums are combined:
      "tree"   (default) -- pairwise tree reduction, ceil(log2(n)) SEQUENTIAL
                levels, each level's additions independent of one another (and
                so, in principle, pipelineable). This is what any reasonable
                hardware/compiler would do, not an optimistic stretch.
      "serial" -- naive sequential chain, n-1 dependent additions. A clearly
                worse upper bound, kept only as a documented sensitivity check
                -- an initial version used this by accident and it dominated
                layer cost by ~94%, which is implausible for a reduction this
                shallow and was the signal something was being modelled too
                pessimistically.
    """
    row_tiles = -(-m.weight_rows // 64)
    bit_slices = -(-VALUE_BITS // CELL_BITS)

    mvm = tile.mvm(
        rows=m.weight_rows, cols=m.weight_cols,
        input_bits=INPUT_BITS, output_bits=OUTPUT_BITS, value_bits=VALUE_BITS,
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
        # Energy charges every pairwise add regardless of reduction shape (the
        # same total work happens either way); only LATENCY differs with depth.
        accumulate_energy = (partials_to_combine - 1) * add.energy_pJ

    return LayerCost(
        name=m.name,
        cycles=int((mvm.cycles + accumulate_cycles) * positions),
        energy_pJ=(mvm.energy_pJ + accumulate_energy) * positions,
        tiles=m.tiles,
    )


# --------------------------------------------------------------------------
# Non-MVM kernels: batch norm, ReLU, residual add, pooling, all on the DCE.
# --------------------------------------------------------------------------

def nonmvm_cost(tile: HybridComputeTile, elements: int, has_bn: bool,
                has_relu: bool, has_residual: bool) -> LayerCost:
    """Per-position cost of the digital ops following a conv layer.

    Batch norm folds to one multiply-add per channel (modelled as ADD, since the
    DCE has no native multiply primitive cheaper than its MUL8 recipe and a
    scale-and-shift is the common inference-time BN folding); ReLU uses the
    recipe book's native RELUv1; a residual connection is one more ADD.
    """
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
) -> tuple[list[LayerCost], list[LayerCost]]:
    """Full-network cost: (per-layer MVM costs, per-layer non-MVM costs).

    Every conv layer except the network's very last one is followed by BN+ReLU;
    the second conv of every BasicBlock additionally gets a residual add before
    its ReLU (the standard post-activation ResNet design this architecture uses).
    """
    tile = tile or HybridComputeTile(ace_params)
    mappings = map_resnet20(value_bits=VALUE_BITS, ace=tile.ace)

    mvm_costs, nonmvm_costs = [], []
    for i, m in enumerate(mappings):
        mvm_costs.append(layer_cost(m, tile, reduction=reduction))

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
    """Convert the network's total array footprint into HCTs (tiles) needed to
    hold it resident. See ARRAYS_PER_ACE for why this is not a 1:1 sum."""
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
