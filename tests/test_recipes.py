"""Validate the OSCAR cost table parsed out of the authors' own simulator.

These costs drive every digital-PUM timing and energy number we produce, so the
parser failing silently would quietly corrupt the whole evaluation.
"""

import pytest

from darthpum import recipes


@pytest.fixture(scope="module")
def table():
    return recipes.load()


def test_all_recipes_present(table):
    assert len(table) == recipes.EXPECTED_RECIPE_COUNT
    # Operations the three workloads actually need.
    for op in ("ADD", "SUB", "XOR", "AND", "OR", "NOR", "MOV", "LSHIFT", "RELUv1"):
        assert op in table, f"{op} missing from the OSCAR recipe book"


def test_add_crosschecks_against_published_racer_numbers(table):
    """[RACER] reports 22 NOR primitives per bit for an add under MAGIC NOR.
    [OSCAR] then claims ~30% better performance by adding a native OR primitive.

    Our parsed OSCAR ADD should therefore use fewer than 22 NOR and at least one
    OR. This is an independent check that the parse is correct -- the numbers come
    from a C++ header, the expectation comes from two published papers.
    """
    add = table["ADD"]
    assert add.mix["NOR"] < 22, "OSCAR ADD should need fewer NOR than MAGIC's 22/bit"
    assert add.mix.get("OR", 0) >= 1, "OSCAR ADD should use the native OR primitive"
    assert add.cycles == 21


def test_cycles_are_positive_and_ordered(table):
    """Sanity: costs are positive, and complex operations cost more than simple ones."""
    for name, rec in table.items():
        assert rec.cycles > 0, f"{name} has no micro-ops"
        assert rec.switching_uops <= rec.cycles

    assert table["MOV"].cycles < table["ADD"].cycles < table["MUL8"].cycles
    assert table["MUL8"].cycles < table["MUL16"].cycles < table["MUL32"].cycles


def test_xor_is_cheaper_than_add(table):
    """XOR is a single-bit Boolean primitive with no carry chain, so it should
    cost well under an ADD -- part of why bitwise-heavy kernels suit digital
    PUM particularly well."""
    assert table["XOR"].cycles < table["ADD"].cycles


def test_control_uops_excluded_from_switching_count(table):
    """Energy must come only from micro-ops that actually drive current.
    LOCK/RELEASE/NOP are pipeline control and must not be charged."""
    mov = table["MOV"]           # LOCK, NOP, NOP, COPY, RELEASE
    assert mov.cycles == 5
    assert mov.switching_uops == 1   # the COPY alone


def test_parser_matches_cache(table):
    """Re-parsing the header must agree with the committed snapshot, so results
    stay reproducible for anyone who clones without running setup.sh."""
    try:
        fresh = recipes.load(refresh=True)
    except FileNotFoundError:
        pytest.skip("MASTODON not cloned; cached snapshot in use")
    assert {k: v.uops for k, v in fresh.items()} == {k: v.uops for k, v in table.items()}
