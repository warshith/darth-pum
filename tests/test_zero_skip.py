"""Zero-skipping: real measured activation sparsity reducing ACE row-driver
energy -- a modest, honestly-measured energy contribution.

The headline risk this module guards against is the mirror image of the ADC
contribution's: a naive back-of-envelope estimate (51% average activation
sparsity -> expect roughly 48% row-driver energy savings) looks far larger
than what the full network actually realizes (~3%), because ADC conversion
and digital partial-sum recombination -- both unaffected by which rows were
zero -- dominate total ACE energy. These tests pin the real, diluted number.
"""

import pytest

from darthpum.hct import HybridComputeTile
from workloads.resnet20.resnet_pum import inference_cost, layer_cost
from workloads.resnet20.mapping import map_resnet20
from workloads.resnet20.sparsity import MEASURED_SPARSITY, layer_skip_fractions


@pytest.fixture
def tile():
    return HybridComputeTile()


# -- the measurement itself ------------------------------------------------

def test_measured_sparsity_is_real_not_a_textbook_guess():
    """19 ReLU sites, each a genuine measured fraction in [0, 1] -- not a
    single assumed constant repeated for every layer."""
    assert len(MEASURED_SPARSITY) == 19
    assert all(0.0 <= s <= 1.0 for s in MEASURED_SPARSITY)
    # Real per-layer variation, not a flat assumed rate -- the whole point of
    # measuring rather than assuming "CNNs are ~50% sparse" as one constant.
    assert max(MEASURED_SPARSITY) - min(MEASURED_SPARSITY) > 0.3


def test_measured_mean_matches_what_was_reported():
    mean = sum(MEASURED_SPARSITY) / len(MEASURED_SPARSITY)
    assert mean == pytest.approx(0.511, abs=0.01)


def test_layer_skip_fractions_cover_every_mapping_layer(tile):
    """Every one of mapping.py's 20 layers must get an entry -- conv1 (the
    stem, consuming the raw image) explicitly at 0.0, not missing."""
    names = [m.name for m in map_resnet20(ace=tile.ace)]
    fractions = layer_skip_fractions()
    assert set(fractions) == set(names)
    assert fractions[names[0]] == 0.0  # conv1: dense image input, no skip


# -- the mechanism: cycles unchanged, energy reduced proportionally --------

def test_zero_skip_never_changes_cycles(tile):
    """Energy-only contribution: skipping a wordline's driver cannot change
    how many cycles the MVM takes, since the ADC still has to convert the
    bitline regardless of which rows drove it."""
    base_mvm, base_nonmvm = inference_cost(tile, zero_skip=False)
    skip_mvm, skip_nonmvm = inference_cost(tile, zero_skip=True)
    base_cycles = sum(c.cycles for c in base_mvm + base_nonmvm)
    skip_cycles = sum(c.cycles for c in skip_mvm + skip_nonmvm)
    assert base_cycles == skip_cycles


def test_zero_skip_reduces_energy(tile):
    base_mvm, base_nonmvm = inference_cost(tile, zero_skip=False)
    skip_mvm, skip_nonmvm = inference_cost(tile, zero_skip=True)
    base_energy = sum(c.energy_pJ for c in base_mvm + base_nonmvm)
    skip_energy = sum(c.energy_pJ for c in skip_mvm + skip_nonmvm)
    assert skip_energy < base_energy


# -- the honest, diluted end-to-end number ----------------------------------

def test_end_to_end_reduction_is_small_not_the_naive_51_percent_estimate():
    """The headline risk: a naive reading of 51% average sparsity would
    suggest roughly 48% energy savings (since Row Periphery is ~94% of a
    single MVM's energy in isolation). The real, network-wide number is
    much smaller once ADC conversion and bit-slice recombination -- both
    completely unaffected by row-skipping -- are accounted for. Pinned here
    so a future change can't silently let the inflated estimate back in."""
    tile = HybridComputeTile()
    base_mvm, base_nonmvm = inference_cost(tile, zero_skip=False)
    skip_mvm, skip_nonmvm = inference_cost(tile, zero_skip=True)
    base_energy = sum(c.energy_pJ for c in base_mvm + base_nonmvm)
    skip_energy = sum(c.energy_pJ for c in skip_mvm + skip_nonmvm)
    reduction_pct = 100 * (1 - skip_energy / base_energy)

    assert 1.0 < reduction_pct < 10.0, (
        f"zero-skip energy reduction {reduction_pct:.1f}% is outside the "
        "expected small-but-real range -- if this grew to look like the "
        "naive ~48% row-driver-only estimate, something upstream silently "
        "stopped diluting correctly (e.g. ADC or accumulate energy no "
        "longer being charged)"
    )


def test_accumulate_energy_is_the_dominant_untouched_cost(tile):
    """Why the reduction is small: for a wide layer, the DIGITAL cost of
    recombining bit-sliced partial sums (untouched by row-skipping) should
    dominate over the ANALOG MVM cost (which row-skipping does reduce)."""
    m = next(m for m in map_resnet20(ace=tile.ace) if m.name == "r3.0.conv2")
    full = layer_cost(m, tile, active_row_fraction=1.0)
    half = layer_cost(m, tile, active_row_fraction=0.5)
    # Energy drops with more skipping, but not by anywhere near 50%, because
    # accumulate_energy (unaffected) is the majority of this layer's cost.
    reduction = 1 - half.energy_pJ / full.energy_pJ
    assert 0 < reduction < 0.25


# -- interface sanity --------------------------------------------------------

def test_active_row_fraction_defaults_to_no_skip(tile):
    """Every call site elsewhere in the project omits active_row_fraction --
    it must default to 1.0 (no skipping), reproducing every number computed
    before this contribution existed."""
    m = next(m for m in map_resnet20(ace=tile.ace) if m.name == "fc")
    default = layer_cost(m, tile)
    explicit = layer_cost(m, tile, active_row_fraction=1.0)
    assert default.energy_pJ == explicit.energy_pJ
    assert default.cycles == explicit.cycles


def test_zero_skip_false_matches_pre_contribution_numbers(tile):
    """zero_skip=False (the default) must reproduce the exact energy figure
    every other committed ResNet-20 result in this project was computed
    against, so this contribution cannot silently change prior results."""
    mvm_costs, nonmvm_costs = inference_cost(tile, zero_skip=False)
    total_energy = sum(c.energy_pJ for c in mvm_costs + nonmvm_costs)
    assert total_energy == pytest.approx(726_837_072, rel=1e-6)
