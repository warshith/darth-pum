"""Hybrid compute tile: the analog/digital coordination layer."""

import pytest

from darthpum.device import ACEParams, ADCParams
from darthpum.hct import HybridComputeTile


@pytest.fixture
def hct():
    return HybridComputeTile(optimized=True)


# interconnect

def test_transfer_respects_the_8_byte_network_width(hct):
    # [DARTH] Sec 4: ACE<->DCE network moves 8 B/cycle
    assert hct.transfer_cycles(values=64, bits_per_value=8) == 8    # 64 B
    assert hct.transfer_cycles(values=64, bits_per_value=1) == 1    # 8 B
    assert hct.transfer_cycles(values=1, bits_per_value=1) == 1     # partial cycle


def test_shift_is_free_only_when_optimized():
    """Shift units act during the transfer, so optimized pays nothing to place
    a partial product at its bit position. Without them it has to walk up the
    pipeline one position per cycle."""
    assert HybridComputeTile(optimized=True).shift_cycles(7) == 0
    assert HybridComputeTile(optimized=False).shift_cycles(7) == 7


# the Figure 10 ablation

def test_optimized_timeline_beats_unoptimized():
    """This should reproduce Figure 10 -- overlapping the shift with the
    transfer and pipelining the adds ought to be worth several times over
    the fully serial version."""
    a = HybridComputeTile(optimized=False).mvm(64, 64, input_bits=8, value_bits=1)
    b = HybridComputeTile(optimized=True).mvm(64, 64, input_bits=8, value_bits=1)
    assert b.cycles < a.cycles
    assert a.cycles / b.cycles > 3


def test_unoptimized_stalls_the_analog_element():
    """[DARTH] Sec 4.1: writes, shifts, adds can't interleave without wrecking
    the bit-pipelined model, so analog just sits idle holding a finished
    partial product with nowhere to go. That stall is exactly what the shift
    units are there to remove."""
    a = HybridComputeTile(optimized=False).mvm(64, 64, input_bits=8, value_bits=1)
    b = HybridComputeTile(optimized=True).mvm(64, 64, input_bits=8, value_bits=1)
    assert a.stall_cycles > 0
    assert a.stall_cycles > b.stall_cycles
    assert a.stall_cycles / a.cycles > 0.5  # most of unoptimized time is pure stall


def test_fewer_input_bits_means_fewer_partial_products(hct):
    """Each input bit-slice is one analog pass + one accumulate, so halving
    input precision should roughly halve the total work."""
    wide = hct.mvm(64, 64, input_bits=8, value_bits=1).cycles
    narrow = hct.mvm(64, 64, input_bits=4, value_bits=1).cycles
    assert narrow < wide


# cost accounting sanity

def test_cycle_accounting_is_consistent(hct):
    c = hct.mvm(64, 64, input_bits=8, value_bits=1)
    assert c.cycles > 0
    assert c.analog_cycles > 0
    assert c.energy_pJ > 0
    assert c.cycles >= c.analog_cycles, "total cannot be less than the analog part"


def test_extra_analog_arrays_are_free_until_the_ace_fills(hct):
    """Arrays run in parallel, so 9 arrays costs the same as 1, as long as
    both fit in one ACE's 64. This is exactly why ResNet-20 (largest layer:
    9 arrays) gets throughput from replicating across tiles, not from any one
    tile running faster."""
    small = hct.mvm(64, 64, input_bits=8, value_bits=1).cycles
    large = hct.mvm(576, 64, input_bits=8, value_bits=1).cycles
    assert small == large


def test_converter_choice_propagates_to_tile_level():
    """The SAR/ramp reversal needs to survive into end-to-end tile cost, not
    just stay true in the isolated analog model -- otherwise the ADC
    comparison wouldn't actually say anything about real workloads."""
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


# digital-only path

def test_digital_only_op_uses_no_analog_cycles(hct):
    # activations, XORs etc. never touch the analog element
    c = hct.digital_op("RELUv1", width_bits=8, elements=64)
    assert c.analog_cycles == 0
    assert c.digital_cycles > 0
    assert c.stall_cycles == 0
    assert c.energy_pJ > 0


def test_frontend_energy_is_amortised_over_eight_tiles(hct):
    """[DARTH] Table 3: one front end shared by 8 HCTs -> each tile pays 1/8
    of the 63 mW."""
    from darthpum.area import FRONTEND_POWER_MW
    cycles = 1000
    expected = (FRONTEND_POWER_MW / 8) * cycles   # 1 GHz -> cycles == ns
    assert hct._frontend_energy_pJ(cycles) == pytest.approx(expected)
