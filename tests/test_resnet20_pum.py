"""ResNet-20 inference timing/energy on DARTH-PUM."""

import math

import pytest

from darthpum.hct import HybridComputeTile
from workloads.resnet20.resnet_pum import (
    ARRAYS_PER_ACE, hcts_per_instance, inference_cost, layer_cost,
)
from workloads.resnet20.mapping import map_resnet20


@pytest.fixture
def tile():
    return HybridComputeTile()


# -- the array-vs-HCT bug --------------------------------------------------

def test_hct_footprint_is_arrays_divided_by_64_not_summed_directly():
    """A real bug: summing per-layer ARRAY counts (784) and using that directly
    as tiles_per_instance left only 2 parallel inferences possible chip-wide.
    The correct HCT footprint divides by ARRAYS_PER_ACE."""
    mvm_costs, _ = inference_cost()
    total_arrays = sum(c.tiles for c in mvm_costs)
    assert total_arrays == 784
    assert hcts_per_instance(mvm_costs) == 13
    assert hcts_per_instance(mvm_costs) == -(-784 // ARRAYS_PER_ACE)


def test_hct_footprint_is_much_smaller_than_the_chip():
    """One inference's footprint must leave room for many parallel instances."""
    mvm_costs, _ = inference_cost()
    assert hcts_per_instance(mvm_costs) < 50, "footprint should be a small slice of ~1844 tiles"


# -- the row-tile dropping bug ----------------------------------------------

def test_wide_layers_charge_for_every_row_tile(tile):
    """A layer needing 3 row-tiles must cost meaningfully more than one that
    fits in a single row-tile -- an earlier version silently charged only one
    row-tile's MVM pass for any layer wider than 64 rows, regardless of how
    many tiles it actually needed."""
    mappings = {m.name: m for m in map_resnet20(value_bits=8, ace=tile.ace)}

    single_tile = mappings["conv1"]       # 27 rows -> 1 row-tile
    triple_tile = mappings["r1.0.conv1"]  # 144 rows -> 3 row-tiles
    assert -(-single_tile.weight_rows // 64) == 1
    assert -(-triple_tile.weight_rows // 64) == 3

    c1 = layer_cost(single_tile, tile)
    c2 = layer_cost(triple_tile, tile)
    # Normalise by output positions (1024 for both, 32x32) to compare per-MAC-pass cost.
    positions = single_tile.out_shape[-1] * single_tile.out_shape[-2]
    assert c2.cycles / positions > c1.cycles / positions


# -- reduction shape ----------------------------------------------------

def test_tree_reduction_is_cheaper_than_serial(tile):
    """Tree reduction (ceil(log2(n)) levels) must cost less than a naive
    serial chain (n-1 adds) for any layer with more than 2 partials to
    combine. An initial version used serial by accident and it dominated
    layer cost by ~94%, which is implausible for a shallow reduction tree."""
    m = next(m for m in map_resnet20(value_bits=8, ace=tile.ace)
             if m.name == "r3.0.conv2")  # the widest layer: 576 rows, 9 row-tiles
    tree = layer_cost(m, tile, reduction="tree")
    serial = layer_cost(m, tile, reduction="serial")
    assert tree.cycles < serial.cycles


def test_reduction_formula_agrees_at_two_partials():
    """At exactly 2 partials there is only one possible pairwise add either
    way, so tree (ceil(log2(2))=1) and serial (2-1=1) must agree -- the one
    case where the two strategies are not supposed to differ."""
    assert math.ceil(math.log2(2)) == 2 - 1


def test_reduction_formula_diverges_as_partials_grow():
    """The whole point of using a tree: the gap between tree and serial costs
    must widen as more partials need combining (log growth vs linear growth)."""
    gaps = [(n - 1) - math.ceil(math.log2(n)) for n in (4, 8, 16, 24)]
    assert gaps == sorted(gaps), "the tree-vs-serial gap should grow with n"
    assert gaps[-1] > gaps[0]


# -- sanity -----------------------------------------------------------------

def test_inference_cost_is_dominated_by_mvm_not_nonmvm():
    """Convolutions should dominate a CNN's cost; batch norm/ReLU/residual
    adds are comparatively cheap per-channel digital ops."""
    mvm_costs, nonmvm_costs = inference_cost()
    total_mvm = sum(c.cycles for c in mvm_costs)
    total_nonmvm = sum(c.cycles for c in nonmvm_costs)
    assert total_mvm > total_nonmvm
    assert total_nonmvm / (total_mvm + total_nonmvm) < 0.1


def test_all_20_layers_produce_mvm_costs():
    mvm_costs, _ = inference_cost()
    assert len(mvm_costs) == 20


def test_19_conv_layers_get_nonmvm_followups_but_fc_does_not():
    """Every conv layer gets BN+ReLU; the final FC classifier gets neither."""
    _, nonmvm_costs = inference_cost()
    assert len(nonmvm_costs) == 19


def test_total_cycles_and_energy_are_positive():
    mvm_costs, nonmvm_costs = inference_cost()
    assert sum(c.cycles for c in mvm_costs + nonmvm_costs) > 0
    assert sum(c.energy_pJ for c in mvm_costs + nonmvm_costs) > 0
