"""Validate the area model against the tile counts the paper states.

This is the project's first and most load-bearing reproduction check: every
throughput number in the paper is an iso-area comparison, so an error in the tile
count propagates straight into every headline result.
"""

import pytest

from darthpum.area import hct_area, hct_count, reconcile
from darthpum.device import ACEParams, ADCParams


def test_sar_tile_count_matches_paper():
    """[DARTH] Section 6: 1860 HCTs with SAR ADCs.

    We get 1844, within 1%.
    """
    n = hct_count(ace=ACEParams(adc=ADCParams.sar()))
    assert n == pytest.approx(1860, rel=0.02), (
        f"SAR tile count {n} is more than 2% from the paper's 1860"
    )


def test_area_components_sum_correctly():
    """The breakdown must add up, and the front end must be amortised over 8 tiles."""
    b = hct_area(ace=ACEParams(adc=ADCParams.sar()))
    assert b.total_um2 == pytest.approx(
        b.dce_um2 + b.ace_um2 + b.glue_um2 + b.adc_um2 + b.frontend_share_um2
    )
    assert b.frontend_share_um2 == pytest.approx(87_000.0 / 8)
    # Two SAR ADCs at 600 um^2 each.
    assert b.adc_um2 == pytest.approx(1_200.0)


def test_ramp_configuration_does_not_reconcile():
    """A known, deliberate discrepancy -- documented rather than fitted away.

    Table 2 lists 1 ramp ADC per analog element. With that, our model puts 1811
    tiles on the die, but the paper states 1660. Since our SAR number reproduces
    to within 1%, the inconsistency is in the ramp configuration, not in the
    model: landing on 1660 requires roughly 4 ramp ADCs per element.

    This test pins the finding. If it ever starts failing, either the model
    changed or the discrepancy was resolved -- both worth noticing.
    """
    r = reconcile()["RAMP"]
    assert r["hcts_model"] > r["hcts_paper"], (
        "expected our ramp tile count to exceed the paper's"
    )
    assert r["implied_adc_count"] == pytest.approx(4.0, abs=0.5), (
        f"implied ramp ADC count {r['implied_adc_count']:.2f} is no longer ~4; "
        "the reported finding needs revisiting"
    )


def test_ramp_tiles_are_larger_than_sar_tiles():
    """A ramp ADC is 3800 um^2 against a SAR's 600, so ramp tiles must be bigger
    and therefore fewer -- the direction the paper also reports."""
    sar = hct_area(ace=ACEParams(adc=ADCParams.sar())).total_um2
    ramp = hct_area(ace=ACEParams(adc=ADCParams.ramp())).total_um2
    assert ramp > sar
