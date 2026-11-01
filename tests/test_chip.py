"""Chip-level scheduling: throughput, thermal limits, and the comparison metrics."""

import math

import pytest

from darthpum.chip import DEFAULT_THERMAL_LIMIT_W, Chip, Workload
from darthpum.device import ACEParams, ADCParams


@pytest.fixture
def chip():
    return Chip()


@pytest.fixture
def tiny():
    """A cheap one-tile workload, chosen so the thermal cap never binds."""
    return Workload(
        name="tiny",
        tiles_per_instance=1,
        cycles_per_instance=1000,
        energy_pJ_per_instance=1.0,
        unit="ops",
    )


# -- budgets ------------------------------------------------------------

def test_tile_count_follows_the_area_model(chip):
    assert chip.tiles == 1844


def test_ramp_chip_holds_fewer_tiles():
    """Ramp converters are 3800 um^2 against a SAR's 600, so they cost tiles."""
    sar = Chip(ace_params=ACEParams(adc=ADCParams.sar())).tiles
    ramp = Chip(ace_params=ACEParams(adc=ADCParams.ramp())).tiles
    assert ramp < sar


# -- throughput ---------------------------------------------------------

def test_throughput_is_parallelism_over_latency(chip, tiny):
    r = chip.run(tiny)
    seconds = tiny.cycles_per_instance / 1e9          # 1 GHz
    assert r.throughput_per_s == pytest.approx(r.instances_parallel / seconds)


def test_footprint_sets_parallelism(chip, tiny):
    """Throughput is as much about how many tiles an instance occupies as about
    how fast it runs -- the argument [DARTH] Section 7.1 makes against AppAccel,
    which spends area on special-function units and so runs fewer instances."""
    one = chip.run(tiny)
    four = chip.run(Workload("wide", 4, tiny.cycles_per_instance,
                             tiny.energy_pJ_per_instance, unit="ops"))
    assert one.instances_parallel == chip.tiles
    assert four.instances_parallel == chip.tiles // 4
    assert one.throughput_per_s > four.throughput_per_s


def test_energy_per_unit_is_independent_of_parallelism(chip, tiny):
    """Running more copies costs more total power but not more energy per result."""
    r = chip.run(tiny)
    assert r.energy_pJ_per_unit == pytest.approx(tiny.energy_pJ_per_instance)


def test_work_units_rescale_throughput_and_energy(chip):
    """A workload reporting 16 blocks per instance should show 16x the throughput
    and a sixteenth of the per-unit energy of one reporting a single block."""
    base = Workload("one", 1, 1000, 160.0, work_units=1, unit="blocks")
    many = Workload("many", 1, 1000, 160.0, work_units=16, unit="blocks")
    a, b = chip.run(base), chip.run(many)
    assert b.throughput_per_s == pytest.approx(16 * a.throughput_per_s)
    assert b.energy_pJ_per_unit == pytest.approx(a.energy_pJ_per_unit / 16)


# -- thermal ------------------------------------------------------------

def test_cheap_workload_is_not_throttled(chip, tiny):
    r = chip.run(tiny)
    assert not r.thermally_limited
    assert r.instances_parallel == chip.tiles
    assert r.power_W < DEFAULT_THERMAL_LIMIT_W


def test_hot_workload_is_throttled_to_the_power_budget(chip):
    """Resistive memory draws real current when it switches, so a chip cannot run
    every array at once. [DARTH] Section 6 notes the DigitalPUM comparison point
    runs only two of its 64 pipelines per cluster for exactly this reason.
    Ignoring the cap would flatter every configuration."""
    hot = Workload("hot", 1, 1000, energy_pJ_per_instance=1e6, unit="ops")
    r = chip.run(hot)
    assert r.thermally_limited
    assert r.instances_parallel < chip.tiles
    assert r.power_W <= DEFAULT_THERMAL_LIMIT_W + 1e-6


def test_raising_the_thermal_limit_lifts_throughput():
    hot = Workload("hot", 1, 1000, energy_pJ_per_instance=1e5, unit="ops")
    cool = Chip(thermal_limit_W=80).run(hot)
    warm = Chip(thermal_limit_W=800).run(hot)
    assert warm.throughput_per_s > cool.throughput_per_s


def test_at_least_one_instance_always_runs(chip):
    """Even a workload over the power budget on its own must not report zero
    throughput -- it runs, just alone."""
    furnace = Workload("furnace", 1, 1000, energy_pJ_per_instance=1e12, unit="ops")
    assert chip.run(furnace).instances_parallel == 1


# -- comparison metrics -------------------------------------------------

def test_speedup_and_energy_saving_are_the_papers_conventions(chip):
    """Both are reported so that larger is better: speedup divides throughputs,
    energy saving divides baseline energy by ours."""
    fast = chip.run(Workload("fast", 1, 500, 10.0, unit="ops"))
    slow = chip.run(Workload("slow", 1, 1000, 20.0, unit="ops"))
    assert Chip.speedup(fast, slow) == pytest.approx(2.0)
    assert Chip.energy_saving(fast, slow) == pytest.approx(2.0)


def test_geomean():
    assert Chip.geomean([2.0, 8.0]) == pytest.approx(4.0)
    assert Chip.geomean([59.4, 14.8, 40.8]) == pytest.approx(
        math.exp((math.log(59.4) + math.log(14.8) + math.log(40.8)) / 3))


def test_geomean_rejects_nonpositive():
    """A zero or negative speedup means an upstream bug; silently returning nan
    would let it reach the report."""
    with pytest.raises(ValueError):
        Chip.geomean([1.0, 0.0])
    with pytest.raises(ValueError):
        Chip.geomean([])


# -- guards -------------------------------------------------------------

def test_workload_larger_than_the_chip_is_rejected(chip):
    """Better to fail loudly than silently report a throughput for something that
    does not fit."""
    with pytest.raises(ValueError, match="needs .* tiles"):
        chip.run(Workload("huge", chip.tiles + 1, 1000, 1.0))


def test_degenerate_workloads_are_rejected():
    with pytest.raises(ValueError, match="at least one tile"):
        Workload("bad", 0, 1000, 1.0)
    with pytest.raises(ValueError, match="at least one cycle"):
        Workload("bad", 1, 0, 1.0)
