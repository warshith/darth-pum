"""Heterogeneous-memory HCT: swap the DCE's device model and see what moves.

Main thing to check: the swap should change energy and endurance (it does
something) without changing anything about cycles, tile counts or any other
already-reported number (it doesn't change anything it has no business
touching).
"""

import pytest

from contribution.heterogeneous_memory import (
    compare_devices, dce_switching_events_per_inference, endurance_margin,
)
from darthpum.dce import DigitalComputeElement
from darthpum.device import DCEParams, DeviceParams


@pytest.fixture(scope="module")
def results():
    return compare_devices()


def test_both_devices_present(results):
    assert set(results) == {"ReRAM", "STT-MRAM"}


def test_mram_has_far_higher_endurance(results):
    rr, mr = results["ReRAM"], results["STT-MRAM"]
    assert mr.endurance_writes > rr.endurance_writes
    assert mr.years_to_wearout > rr.years_to_wearout


def test_mram_switch_energy_is_higher_not_lower(results):
    # no free lunch -- MRAM's endurance edge doesn't also mean it's cheaper
    # to switch, and the module docstring is explicit this is a real tradeoff
    rr, mr = results["ReRAM"], results["STT-MRAM"]
    assert mr.switch_energy_pJ > rr.switch_energy_pJ


def test_switching_event_count_is_device_independent(results):
    # same ResNet-20 non-MVM op mix runs on either device -- only the per-
    # event energy/endurance interpretation should differ, not the count
    rr, mr = results["ReRAM"], results["STT-MRAM"]
    assert rr.switching_events_per_inference == mr.switching_events_per_inference


def test_endurance_ratio_matches_device_parameter_ratio(results):
    # same wear pattern on both, so inferences-to-wearout should scale with
    # exactly the endurance_writes ratio we put into DeviceParams.stt_mram()
    rr, mr = results["ReRAM"], results["STT-MRAM"]
    expected = mr.endurance_writes / rr.endurance_writes
    actual = mr.inferences_to_wearout / rr.inferences_to_wearout
    assert actual == pytest.approx(expected, rel=1e-9)


def test_dce_switching_energy_responds_to_its_own_device():
    # this is the actual code change the extension hinges on: dce.py reads
    # self.p.device, not a fixed module-level constant
    reram_dce = DigitalComputeElement(DCEParams(device=DeviceParams()))
    mram_dce = DigitalComputeElement(DCEParams(device=DeviceParams.stt_mram()))
    reram_e = reram_dce.switching_energy_pJ("ADD", width_bits=8, elements=64)
    mram_e = mram_dce.switching_energy_pJ("ADD", width_bits=8, elements=64)
    assert mram_e != reram_e
    assert mram_e == pytest.approx(
        reram_e * DeviceParams.stt_mram().switch_energy_pJ / DeviceParams().switch_energy_pJ
    )


def test_switching_events_scale_linearly_with_width():
    # sanity: directly counting switching_events for one op scales linearly
    # in width_bits, since that's just the definition
    dce = DigitalComputeElement(DCEParams())
    e8 = dce.switching_events("ADD", width_bits=8, elements=64)
    e16 = dce.switching_events("ADD", width_bits=16, elements=64)
    assert e16 == 2 * e8


def test_per_inference_count_is_positive_and_reproducible():
    dce = DigitalComputeElement(DCEParams())
    a = dce_switching_events_per_inference(dce)
    b = dce_switching_events_per_inference(dce)
    assert a > 0
    assert a == b


def test_ace_params_default_is_untouched_by_this_extension():
    # the ACE isn't part of this swap -- it stays ReRAM, always. Nothing here
    # should have given ACEParams a device knob at all.
    from darthpum.device import ACEParams

    assert not hasattr(ACEParams(), "device")
