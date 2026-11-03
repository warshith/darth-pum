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
    """Cheap one-tile workload -- picked so the thermal cap never kicks in."""
    return Workload(
        name="tiny",
        tiles_per_instance=1,
        cycles_per_instance=1000,
        energy_pJ_per_instance=1.0,
        unit="ops",
    )


def test_tile_count_follows_the_area_model(chip):
    assert chip.tiles == 1844


def test_ramp_chip_holds_fewer_tiles():
    """Ramp ADCs are 3800 um^2 against a SAR's 600, so they eat into tile count."""
    sar = Chip(ace_params=ACEParams(adc=ADCParams.sar())).tiles
    ramp = Chip(ace_params=ACEParams(adc=ADCParams.ramp())).tiles
    assert ramp < sar


def test_throughput_is_parallelism_over_latency(chip, tiny):
    r = chip.run(tiny)
    seconds = tiny.cycles_per_instance / 1e9  # 1 GHz
    assert r.throughput_per_s == pytest.approx(r.instances_parallel / seconds)


def test_footprint_sets_parallelism(chip, tiny):
    """Throughput depends on tile footprint just as much as raw speed -- the
    argument [DARTH] 7.1 makes against AppAccel, which burns area on
    special-function units and ends up running fewer instances at once."""
    one = chip.run(tiny)
    four = chip.run(Workload("wide", 4, tiny.cycles_per_instance,
                             tiny.energy_pJ_per_instance, unit="ops"))
    assert one.instances_parallel == chip.tiles
    assert four.instances_parallel == chip.tiles // 4
    assert one.throughput_per_s > four.throughput_per_s


def test_energy_per_unit_is_independent_of_parallelism(chip, tiny):
    """More copies running = more total power, not more energy per result."""
    r = chip.run(tiny)
    assert r.energy_pJ_per_unit == pytest.approx(tiny.energy_pJ_per_instance)


def test_work_units_rescale_throughput_and_energy(chip):
    """16 blocks/instance should give 16x the throughput and 1/16th the
    per-unit energy of a workload reporting just 1 block/instance."""
    base = Workload("one", 1, 1000, 160.0, work_units=1, unit="blocks")
    many = Workload("many", 1, 1000, 160.0, work_units=16, unit="blocks")
    a, b = chip.run(base), chip.run(many)
    assert b.throughput_per_s == pytest.approx(16 * a.throughput_per_s)
    assert b.energy_pJ_per_unit == pytest.approx(a.energy_pJ_per_unit / 16)


def test_cheap_workload_is_not_throttled(chip, tiny):
    r = chip.run(tiny)
    assert not r.thermally_limited
    assert r.instances_parallel == chip.tiles
    assert r.power_W < DEFAULT_THERMAL_LIMIT_W


def test_hot_workload_is_throttled_to_the_power_budget(chip):
    """ReRAM draws real current switching, so you can't run every array at
    once. The paper's DigitalPUM comparison runs only 2 of 64 pipelines per
    cluster for exactly this reason -- skip the cap and everything looks
    better than it should."""
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
    """Even something way over the power budget shouldn't report zero
    throughput -- it still runs, just by itself."""
    furnace = Workload("furnace", 1, 1000, energy_pJ_per_instance=1e12, unit="ops")
    assert chip.run(furnace).instances_parallel == 1


def test_speedup_and_energy_saving_are_the_papers_conventions(chip):
    """Both reported so bigger is better: speedup divides throughputs,
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
    """A zero/negative speedup means something upstream is broken -- better to
    raise than silently hand back a nan that reaches the report."""
    with pytest.raises(ValueError):
        Chip.geomean([1.0, 0.0])
    with pytest.raises(ValueError):
        Chip.geomean([])


def test_workload_larger_than_the_chip_is_rejected(chip):
    with pytest.raises(ValueError, match="needs .* tiles"):
        chip.run(Workload("huge", chip.tiles + 1, 1000, 1.0))


def test_degenerate_workloads_are_rejected():
    with pytest.raises(ValueError, match="at least one tile"):
        Workload("bad", 0, 1000, 1.0)
    with pytest.raises(ValueError, match="at least one cycle"):
        Workload("bad", 1, 0, 1.0)
