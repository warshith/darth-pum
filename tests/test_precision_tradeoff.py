"""Precision tradeoff: throughput scaling vs. measured accuracy cost as each
analog cell stores more bits.

Main thing to check: unlike zero-skipping, this gain shouldn't get diluted --
every layer's footprint shrinks by the same factor no matter what else is
going on, so the full tile-count ratio should show up at the chip level too.
"""

import pytest

from contribution.precision_tradeoff import (
    CELL_BITS_SWEPT, MEASURED_ACCURACY_PCT, throughput_sweep,
)


@pytest.fixture(scope="module")
def points():
    return throughput_sweep()


def test_sweep_covers_all_three_cell_bit_settings(points):
    assert [p.cell_bits for p in points] == list(CELL_BITS_SWEPT)


def test_more_bits_per_cell_needs_fewer_tiles(points):
    # doubling bits/cell should roughly halve tile count, since
    # tiles_for_matrix just divides by cell_bits
    tiles = {p.cell_bits: p.tiles_per_instance for p in points}
    assert tiles[2] < tiles[1]
    assert tiles[4] < tiles[2]


def test_throughput_increases_with_bits_per_cell(points):
    throughput = {p.cell_bits: p.chip_throughput_per_s for p in points}
    assert throughput[2] > throughput[1]
    assert throughput[4] > throughput[2]


def test_throughput_gain_is_not_diluted(points):
    # this is the thing that makes this contribution worth picking over a
    # weaker one -- 2 bits/cell should land near a full 2x, not get chewed
    # down to a few percent the way zero-skip's energy number did
    base = next(p for p in points if p.cell_bits == 1)
    two_bit = next(p for p in points if p.cell_bits == 2)
    speedup = two_bit.chip_throughput_per_s / base.chip_throughput_per_s
    assert speedup > 1.5, (
        f"2-bit/cell speedup only {speedup:.2f}x -- looks diluted, which "
        "would undermine the whole point of this contribution"
    )


def test_accuracy_decreases_with_bits_per_cell():
    # more levels per cell = harder to tell apart = more programming error.
    # should fall monotonically, no exceptions
    assert MEASURED_ACCURACY_PCT[1] > MEASURED_ACCURACY_PCT[2] > MEASURED_ACCURACY_PCT[4]


def test_slc_accuracy_is_close_to_the_papers_own_slc_result():
    # cell_bits=1 here is the same CrossSim config as the paper's own SLC
    # number elsewhere in the project (workloads/resnet20/accuracy.py), so
    # these shouldn't just be in the same ballpark, they should basically match
    assert MEASURED_ACCURACY_PCT[1] == pytest.approx(83.43, abs=0.5)


def test_four_bit_accuracy_cost_is_large_not_negligible():
    # honest finding here: more bits/cell is not a free lunch. make sure the
    # 1-bit -> 4-bit drop is big enough that nobody reads this as a win with
    # no downside
    drop = MEASURED_ACCURACY_PCT[1] - MEASURED_ACCURACY_PCT[4]
    assert drop > 15.0
