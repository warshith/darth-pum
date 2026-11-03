"""Baseline tests: capacity vs latency, round-trip granularity, and whether
the calibrated model actually reproduces the paper's ratios."""

import pytest

from baselines.resnet20_baseline import (
    CALIBRATED_LINK_GB_PER_S, CALIBRATED_ROUND_TRIP_NS, ResNet20Baseline,
)
from darthpum.chip import Chip, Workload
from darthpum.hct import HybridComputeTile
from workloads.resnet20.resnet_pum import hcts_per_instance, inference_cost


@pytest.fixture(scope="module")
def tile():
    return HybridComputeTile()


@pytest.fixture(scope="module")
def darthpum_numbers(tile):
    mvm_costs, nonmvm_costs = inference_cost(tile)
    total_cycles = sum(c.cycles for c in mvm_costs + nonmvm_costs)
    total_energy = sum(c.energy_pJ for c in mvm_costs + nonmvm_costs)
    w = Workload("ResNet-20", hcts_per_instance(mvm_costs), total_cycles,
                total_energy, unit="inferences")
    r = Chip().run(w)
    return r.throughput_per_s, total_cycles / 1000  # (img/s, latency us)


# -- accelerator capacity (first pass treated this as one image at a time) --

def test_accelerator_holds_thousands_of_parallel_copies(tile):
    """A 1.5 GB accelerator running a 269K-param (784-array) network should
    obviously run way more than one image at a time -- first draft didn't,
    and the Baseline came out at a laughable 73.5 img/s."""
    b = ResNet20Baseline(tile=tile)
    assert b.accelerator_parallel_capacity() > 1000


def test_accelerator_capacity_scales_with_accelerator_size(tile):
    small = ResNet20Baseline(tile=tile, accelerator_GB=0.1).accelerator_parallel_capacity()
    large = ResNet20Baseline(tile=tile, accelerator_GB=1.5).accelerator_parallel_capacity()
    assert large > small


# -- round-trip granularity --------------------------------------------------

def test_roundtrips_counted_per_array_not_per_layer(tile):
    """Round trips are per physical array (row-tile x bit-slice), not per
    layer. Per-layer needed an implausible ~444 us/round-trip to match the
    paper's latency ratio; per-array only needs ~11 us, which is a normal
    number for device I/O."""
    b = ResNet20Baseline(tile=tile)
    total_roundtrips = sum(b.roundtrips_per_layer(m) for m in b.mappings)
    assert total_roundtrips == 784          # one per physical array
    assert total_roundtrips > 20 * 5        # meaningfully more than per-layer


def test_each_layers_roundtrip_count_matches_its_array_footprint(tile):
    """roundtrips_per_layer and LayerMapping.tiles count arrays the same way
    -- should agree per layer, not just in the total."""
    b = ResNet20Baseline(tile=tile)
    for m in b.mappings:
        assert b.roundtrips_per_layer(m) == m.tiles


# -- capacity and latency are not the same thing -----------------------------

def test_latency_uses_one_core_capacity_uses_all(tile):
    """One image's critical path runs on a single core regardless of how many
    you have. Throughput does scale with cores since images don't depend on
    each other."""
    b2 = ResNet20Baseline(tile=tile, cores=2)
    b8 = ResNet20Baseline(tile=tile, cores=8)
    assert b2.run().single_inference_latency_us == pytest.approx(
        b8.run().single_inference_latency_us
    )
    assert b8.cpu_capacity_images_per_s() > b2.cpu_capacity_images_per_s()


def test_bandwidth_capacity_is_shared_across_concurrent_images(tile):
    """Bandwidth caps throughput (aggregate bytes/s across every in-flight
    image), it's not a per-image latency thing -- so halving the link should
    just halve this capacity."""
    fast = ResNet20Baseline(tile=tile, link_GB_per_s=50.0)
    slow = ResNet20Baseline(tile=tile, link_GB_per_s=25.0)
    assert slow.transfer_bandwidth_capacity_images_per_s() == pytest.approx(
        fast.transfer_bandwidth_capacity_images_per_s() / 2
    )


# -- calibrated reproduction --------------------------------------------------

def test_calibrated_model_reproduces_the_papers_throughput_ratio(tile, darthpum_numbers):
    dp_throughput, _ = darthpum_numbers
    b = ResNet20Baseline(tile=tile, round_trip_latency_ns=CALIBRATED_ROUND_TRIP_NS,
                         link_GB_per_s=CALIBRATED_LINK_GB_PER_S)
    speedup = dp_throughput / b.run().throughput_inferences_per_s
    assert speedup == pytest.approx(14.8, rel=0.02)


def test_calibrated_model_reproduces_the_papers_latency_ratio(tile, darthpum_numbers):
    _, dp_latency_us = darthpum_numbers
    b = ResNet20Baseline(tile=tile, round_trip_latency_ns=CALIBRATED_ROUND_TRIP_NS,
                         link_GB_per_s=CALIBRATED_LINK_GB_PER_S)
    speedup = b.run().single_inference_latency_us / dp_latency_us
    assert speedup == pytest.approx(1.667, rel=0.02)


def test_calibrated_round_trip_latency_is_ordinary():
    """This is really the whole point of calibrating against two ratios at
    once -- the resulting number has to be a normal one, not something that
    only "matches" because it was forced there. 11.3 us is a completely
    ordinary round trip for device I/O."""
    round_trip_us = CALIBRATED_ROUND_TRIP_NS / 1000
    assert 1.0 < round_trip_us < 100.0


def test_calibrated_bandwidth_is_a_plausible_real_interconnect():
    """Same idea -- 4.2 GB/s should sit in normal PCIe/USB territory, not
    some absurd value that only works because it was forced to."""
    assert 0.5 < CALIBRATED_LINK_GB_PER_S < 50


def test_uncalibrated_and_calibrated_give_different_answers(tile, darthpum_numbers):
    """If these two configs happened to agree, the calibration wouldn't
    actually be doing anything."""
    dp_throughput, _ = darthpum_numbers
    uncal = ResNet20Baseline(tile=tile)
    cal = ResNet20Baseline(tile=tile, round_trip_latency_ns=CALIBRATED_ROUND_TRIP_NS,
                           link_GB_per_s=CALIBRATED_LINK_GB_PER_S)
    assert (uncal.run().throughput_inferences_per_s
            != cal.run().throughput_inferences_per_s)
