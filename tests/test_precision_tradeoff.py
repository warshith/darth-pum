"""The precision tradeoff contribution: throughput scaling vs. measured accuracy
cost as cells-per-array store more bits.

Unlike the zero-skipping contribution, this effect should NOT be diluted --
every layer's array footprint shrinks by the same factor regardless of what
else the network is doing, so the full throughput gain should reach the
chip-wide number. These tests check that directly.
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
    """Doubling bits/cell should roughly halve the array footprint, since
    tiles_for_matrix divides by cell_bits directly."""
    tiles = {p.cell_bits: p.tiles_per_instance for p in points}
    assert tiles[2] < tiles[1]
    assert tiles[4] < tiles[2]


def test_throughput_increases_with_bits_per_cell(points):
    throughput = {p.cell_bits: p.chip_throughput_per_s for p in points}
    assert throughput[2] > throughput[1]
    assert throughput[4] > throughput[2]


def test_throughput_gain_is_not_diluted(points):
    """The headline risk this contribution avoided, unlike zero-skipping: the
    full tile-count ratio should show up almost exactly in chip throughput,
    since nothing else in the pipeline depends on cell_bits. 2 bits/cell
    should be close to a full 2x, not diluted down to a few percent."""
    base = next(p for p in points if p.cell_bits == 1)
    two_bit = next(p for p in points if p.cell_bits == 2)
    speedup = two_bit.chip_throughput_per_s / base.chip_throughput_per_s
    assert speedup > 1.5, (
        f"2-bit/cell speedup {speedup:.2f}x is much smaller than expected -- "
        "if this effect is being diluted the way zero-skip's energy saving "
        "was, that would undermine the reason this contribution was chosen"
    )


def test_accuracy_decreases_with_bits_per_cell():
    """More levels per cell means harder-to-distinguish states and more
    programming error -- accuracy must fall monotonically."""
    assert MEASURED_ACCURACY_PCT[1] > MEASURED_ACCURACY_PCT[2] > MEASURED_ACCURACY_PCT[4]


def test_slc_accuracy_is_close_to_the_papers_own_slc_result():
    """cell_bits=1 here must match the SLC accuracy measured elsewhere in this
    project (workloads/resnet20/accuracy.py) -- both come from the identical
    CrossSim configuration, so they should agree closely, not just coincide
    in spirit."""
    assert MEASURED_ACCURACY_PCT[1] == pytest.approx(83.43, abs=0.5)


def test_four_bit_accuracy_cost_is_large_not_negligible():
    """The honest finding: packing more bits per cell is not a free lunch.
    The drop from 1-bit to 4-bit should be large enough that no one could
    mistake this for an unqualified win."""
    drop = MEASURED_ACCURACY_PCT[1] - MEASURED_ACCURACY_PCT[4]
    assert drop > 15.0
