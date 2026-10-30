"""The ResNet-20 Baseline: capacity vs latency, round-trip granularity, and the
calibrated reproduction of the paper's own ratios."""

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


# -- the accelerator-capacity bug (first attempt: zero parallelism) --------

def test_accelerator_holds_thousands_of_parallel_copies(tile):
    """A 1.5 GB accelerator and a 269K-parameter (784-array) network must
    support real parallelism -- treating it as one-image-at-a-time (a first
    attempt's mistake) made the Baseline absurdly slow (73.5 img/s)."""
    b = ResNet20Baseline(tile=tile)
    assert b.accelerator_parallel_capacity() > 1000


def test_accelerator_capacity_scales_with_accelerator_size(tile):
    small = ResNet20Baseline(tile=tile, accelerator_GB=0.1).accelerator_parallel_capacity()
    large = ResNet20Baseline(tile=tile, accelerator_GB=1.5).accelerator_parallel_capacity()
    assert large > small


# -- the round-trip granularity fix -----------------------------------------

def test_roundtrips_counted_per_array_not_per_layer(tile):
    """One round trip per physical array (row-tile x bit-slice combination),
    not one per layer -- a first attempt used per-layer round trips and needed
    an implausible ~444 us/round-trip to explain the paper's latency ratio;
    the array-level granularity needs only ~11 us, an ordinary order of
    magnitude for OS-mediated device I/O."""
    b = ResNet20Baseline(tile=tile)
    total_roundtrips = sum(b.roundtrips_per_layer(m) for m in b.mappings)
    assert total_roundtrips == 784          # one per physical array
    assert total_roundtrips > 20 * 5        # meaningfully more than per-layer


def test_each_layers_roundtrip_count_matches_its_array_footprint(tile):
    """roundtrips_per_layer and LayerMapping.tiles both count physical arrays
    the same way -- they must agree for every layer, not just in total."""
    b = ResNet20Baseline(tile=tile)
    for m in b.mappings:
        assert b.roundtrips_per_layer(m) == m.tiles


# -- capacity vs latency are genuinely different quantities -----------------

def test_latency_uses_one_core_capacity_uses_all(tile):
    """A single image's own critical path runs on one core (no benefit from
    others); THROUGHPUT scales across cores since images are independent, so
    many images can pipeline behind each other's round trips."""
    b2 = ResNet20Baseline(tile=tile, cores=2)
    b8 = ResNet20Baseline(tile=tile, cores=8)
    assert b2.run().single_inference_latency_us == pytest.approx(
        b8.run().single_inference_latency_us
    )
    assert b8.cpu_capacity_images_per_s() > b2.cpu_capacity_images_per_s()


def test_bandwidth_capacity_is_shared_across_concurrent_images(tile):
    """Bandwidth is a THROUGHPUT ceiling (aggregate bytes/s across every
    in-flight image), not a per-image latency term -- halving the link must
    halve the bandwidth-derived capacity."""
    fast = ResNet20Baseline(tile=tile, link_GB_per_s=50.0)
    slow = ResNet20Baseline(tile=tile, link_GB_per_s=25.0)
    assert slow.transfer_bandwidth_capacity_images_per_s() == pytest.approx(
        fast.transfer_bandwidth_capacity_images_per_s() / 2
    )


# -- the calibrated reproduction --------------------------------------------

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
    """The intellectual-honesty check this whole calibration rests on: the
    value must be ordinary, not a number that only works because it was
    forced. 11.3 us is squarely in the range for normal, non-kernel-bypass
    OS-mediated device I/O -- not a negligible latency and not an absurdly
    large one."""
    round_trip_us = CALIBRATED_ROUND_TRIP_NS / 1000
    assert 1.0 < round_trip_us < 100.0


def test_calibrated_bandwidth_is_a_plausible_real_interconnect():
    """4.2 GB/s should land in normal PCIe/USB-class territory, not an
    absurdly fast or slow value that only 'works' because it was forced."""
    assert 0.5 < CALIBRATED_LINK_GB_PER_S < 50


def test_uncalibrated_and_calibrated_give_different_answers(tile, darthpum_numbers):
    """Sanity: the two configurations must not coincidentally agree -- if they
    did, the calibration would not be doing anything."""
    dp_throughput, _ = darthpum_numbers
    uncal = ResNet20Baseline(tile=tile)
    cal = ResNet20Baseline(tile=tile, round_trip_latency_ns=CALIBRATED_ROUND_TRIP_NS,
                           link_GB_per_s=CALIBRATED_LINK_GB_PER_S)
    assert (uncal.run().throughput_inferences_per_s
            != cal.run().throughput_inferences_per_s)
