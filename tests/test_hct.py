"""Hybrid compute tile: the analog/digital coordination layer."""

import pytest

from darthpum.device import ACEParams, ADCParams
from darthpum.hct import HybridComputeTile


@pytest.fixture
def hct():
    return HybridComputeTile(optimized=True)


# -- interconnect -------------------------------------------------------

def test_transfer_respects_the_8_byte_network_width(hct):
    """[DARTH] Section 4: the ACE<->DCE network moves 8 B per cycle."""
    assert hct.transfer_cycles(values=64, bits_per_value=8) == 8    # 64 B
    assert hct.transfer_cycles(values=64, bits_per_value=1) == 1    # 8 B
    assert hct.transfer_cycles(values=1, bits_per_value=1) == 1     # partial cycle


def test_shift_is_free_only_when_optimized():
    """The shift units act during the transfer, so an optimised tile pays nothing
    for placing a partial product at its bit position. Without them the data must
    walk up the pipeline one position per cycle."""
    assert HybridComputeTile(optimized=True).shift_cycles(7) == 0
    assert HybridComputeTile(optimized=False).shift_cycles(7) == 7


# -- the Figure 10 ablation ---------------------------------------------

def test_optimized_timeline_beats_unoptimized():
    """Reproduces Figure 10: overlapping the shift with the transfer, and
    pipelining the adds, should be worth several times the serialised version."""
    a = HybridComputeTile(optimized=False).mvm(64, 64, input_bits=8, value_bits=1)
    b = HybridComputeTile(optimized=True).mvm(64, 64, input_bits=8, value_bits=1)
    assert b.cycles < a.cycles
    assert a.cycles / b.cycles > 3


def test_unoptimized_stalls_the_analog_element():
    """[DARTH] Section 4.1: writes, shifts and adds cannot be interleaved without
    disrupting the bit-pipelined execution model, so the analog element sits idle
    with a finished partial product and nowhere to put it. That stall is the
    bottleneck the shift units exist to remove."""
    a = HybridComputeTile(optimized=False).mvm(64, 64, input_bits=8, value_bits=1)
    b = HybridComputeTile(optimized=True).mvm(64, 64, input_bits=8, value_bits=1)
    assert a.stall_cycles > 0
    assert a.stall_cycles > b.stall_cycles
    # Most of the unoptimised time is stall, not useful work.
    assert a.stall_cycles / a.cycles > 0.5


def test_fewer_input_bits_means_fewer_partial_products(hct):
    """Each input bit-slice costs one analog pass plus one accumulate, so halving
    the input precision should roughly halve the work."""
    wide = hct.mvm(64, 64, input_bits=8, value_bits=1).cycles
    narrow = hct.mvm(64, 64, input_bits=4, value_bits=1).cycles
    assert narrow < wide


# -- cost accounting ----------------------------------------------------

def test_cycle_accounting_is_consistent(hct):
    c = hct.mvm(64, 64, input_bits=8, value_bits=1)
    assert c.cycles > 0
    assert c.analog_cycles > 0
    assert c.energy_pJ > 0
    assert c.cycles >= c.analog_cycles, "total cannot be less than the analog part"


def test_extra_analog_arrays_are_free_until_the_ace_fills(hct):
    """Arrays operate in parallel, so a matrix needing 9 of the ACE's 64 arrays
    costs the same as one needing 1. This is why ResNet-20, whose largest layer is
    9 arrays, gets its throughput from replication across tiles rather than from
    any single tile going faster."""
    small = hct.mvm(64, 64, input_bits=8, value_bits=1).cycles
    large = hct.mvm(576, 64, input_bits=8, value_bits=1).cycles
    assert small == large


def test_converter_choice_propagates_to_tile_level():
    """The SAR/ramp reversal must survive into end-to-end tile cost, not just the
    isolated analog model -- otherwise the ADC study would say nothing about real
    workloads."""
    def tile(adc):
        return HybridComputeTile(ace_params=ACEParams(adc=adc), optimized=True)

    # A 2-output-bit kernel.
    sar2 = tile(ADCParams.sar()).mvm(64, 64, 8, output_bits=2, value_bits=1).cycles
    ramp2 = tile(ADCParams.ramp()).mvm(64, 64, 8, output_bits=2, value_bits=1).cycles
    assert ramp2 < sar2

    # ResNet-20: 8 output bits.
    sar8 = tile(ADCParams.sar()).mvm(64, 64, 8, output_bits=8, value_bits=1).cycles
    ramp8 = tile(ADCParams.ramp()).mvm(64, 64, 8, output_bits=8, value_bits=1).cycles
    assert sar8 < ramp8


# -- digital-only path --------------------------------------------------

def test_digital_only_op_uses_no_analog_cycles(hct):
    """Activations, XORs and the like never touch the analog element."""
    c = hct.digital_op("RELUv1", width_bits=8, elements=64)
    assert c.analog_cycles == 0
    assert c.digital_cycles > 0
    assert c.stall_cycles == 0
    assert c.energy_pJ > 0


def test_frontend_energy_is_amortised_over_eight_tiles(hct):
    """[DARTH] Table 3: one front end is shared by 8 HCTs, so each tile is charged
    an eighth of its 63 mW."""
    from darthpum.area import FRONTEND_POWER_MW
    cycles = 1000
    expected = (FRONTEND_POWER_MW / 8) * cycles   # 1 GHz -> cycles == ns
    assert hct._frontend_energy_pJ(cycles) == pytest.approx(expected)
