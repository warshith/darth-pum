"""Analog compute element: correctness of the dataflow and of the ADC trade-off."""

import numpy as np
import pytest

from darthpum.ace import AnalogComputeElement
from darthpum.device import ACEParams, ADCParams


@pytest.fixture
def ace():
    return AnalogComputeElement()


# functional correctness

def test_bitsliced_mvm_equals_direct_product(ace):
    """This is THE load-bearing check for the analog dataflow -- if bit-sliced
    reconstruction doesn't match the plain product exactly, every analog
    number downstream is wrong. No tolerance, exact integers only."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.integers(0, 16, size=(64, 64))
        v = rng.integers(0, 256, size=64)
        assert np.array_equal(ace.mvm_bitsliced(m, v, input_bits=8), m @ v)


def test_bitsliced_mvm_truncates_when_given_too_few_input_bits(ace):
    """Slicing fewer bits than the input has should drop the high bits, not
    quietly give the full answer anyway (that would hide real precision loss)."""
    m = np.ones((4, 4), dtype=np.int64)
    v = np.array([255, 0, 0, 0])
    got = ace.mvm_bitsliced(m, v, input_bits=4)     # keeps only the low nibble
    assert np.array_equal(got, np.full(4, 15))


def test_mvm_rejects_shape_mismatch(ace):
    with pytest.raises(ValueError, match="shape mismatch"):
        ace.mvm(np.ones((4, 8)), np.ones(4))


# mapping

def test_tile_count_follows_the_isaac_puma_formula(ace):
    # ceil(rows/64) * ceil(cols/64) * ceil(value_bits/cell_bits)
    assert ace.tiles_for_matrix(64, 64, value_bits=1) == 1
    assert ace.tiles_for_matrix(64, 64, value_bits=8) == 8     # 8 bit-slices, SLC
    assert ace.tiles_for_matrix(128, 64, value_bits=1) == 2    # two row-tiles
    assert ace.tiles_for_matrix(65, 65, value_bits=1) == 4     # padding rounds up


def test_resnet20_largest_layer_fits_one_ace():
    """ResNet-20's widest im2col matrix is 576x64 (the 3x3, 64->64 layer3
    convs). SLC -> 9 arrays, nowhere near one ACE's 64, which is exactly why
    the network maps onto a couple of tiles and gets its throughput from
    replication instead of needing to split any single layer."""
    ace = AnalogComputeElement()
    assert ace.tiles_for_matrix(576, 64, value_bits=1) == 9
    assert ace.fits(576, 64, value_bits=1)


# the ADC trade-off

def test_sar_cost_is_independent_of_required_precision():
    """SAR is binary search, one bitline at a time -- cost tracks bitline
    count, not how many output bits you asked for."""
    ace = AnalogComputeElement(ACEParams(adc=ADCParams.sar()))
    assert (ace.adc_cycles_per_slice(2)
            == ace.adc_cycles_per_slice(8)
            == 64 // 2)        # 64 bitlines across 2 ADCs


def test_ramp_cost_grows_exponentially_with_precision():
    """Ramp sweeps 2^bits increments so precision gets expensive fast, but it
    can bail out early when fewer output bits are actually needed."""
    ace = AnalogComputeElement(ACEParams(adc=ADCParams.ramp()))
    assert ace.adc_cycles_per_slice(2) == 4
    assert ace.adc_cycles_per_slice(4) == 16
    assert ace.adc_cycles_per_slice(8) == 256
    assert ace.adc_cycles_per_slice(16) == 256  # never worse than a full sweep


def test_converter_crossover_matches_the_paper():
    """[DARTH] Sec 7.3: SAR wins overall, except kernels that only need a few
    output bits, where ramp's early-termination flips it. Model has to show
    that same reversal, not just pick one winner."""
    sar = AnalogComputeElement(ACEParams(adc=ADCParams.sar()))
    ramp = AnalogComputeElement(ACEParams(adc=ADCParams.ramp()))

    def cycles(a, ob):
        return a.mvm_cost(64, 64, input_bits=8, output_bits=ob, value_bits=1).cycles

    # A 2-output-bit kernel: ramp should win decisively.
    assert cycles(ramp, 2) < cycles(sar, 2)
    assert cycles(sar, 2) / cycles(ramp, 2) > 5

    # ResNet-20: 8 output bits -> SAR should win decisively.
    assert cycles(sar, 8) < cycles(ramp, 8)
    assert cycles(ramp, 8) / cycles(sar, 8) > 5


def test_conversion_dominates_analog_cost(ace):
    """ADCs are the real bottleneck in analog PUM ([DARTH] 2.2.1, 7.3) -- if
    conversion were only a small slice of the cost, the whole ADC comparison
    wouldn't matter."""
    c = ace.mvm_cost(64, 64, input_bits=8, output_bits=8, value_bits=1)
    assert c.adc_fraction > 0.9


def test_more_arrays_than_an_ace_holds_forces_extra_waves(ace):
    """Needing more arrays than one ACE has means extra sequential waves,
    costing proportionally more."""
    one = ace.mvm_cost(64, 64, value_bits=1).cycles
    over = ace.mvm_cost(64 * 65, 64, value_bits=1).cycles   # 65 tiles > 64 arrays
    assert over == 2 * one


def test_unknown_adc_kind_is_rejected():
    ace = AnalogComputeElement(ACEParams(adc=ADCParams(kind="FLASH")))
    with pytest.raises(ValueError, match="unknown ADC kind"):
        ace.adc_cycles_per_slice(8)
