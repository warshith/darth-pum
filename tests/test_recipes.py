"""Validate the OSCAR cost table parsed out of the authors' own simulator.

Every digital-PUM timing/energy number we produce traces back to this table,
so a silent parser failure would quietly corrupt the entire evaluation -- not
a risk worth leaving unchecked.
"""

import pytest

from darthpum import recipes


@pytest.fixture(scope="module")
def table():
    return recipes.load()


def test_all_recipes_present(table):
    assert len(table) == recipes.EXPECTED_RECIPE_COUNT
    for op in ("ADD", "SUB", "XOR", "AND", "OR", "NOR", "MOV", "LSHIFT", "RELUv1"):
        assert op in table, f"{op} missing from the OSCAR recipe book"


def test_add_crosschecks_against_published_racer_numbers(table):
    """[RACER] says an add costs 22 NOR primitives/bit under MAGIC NOR.
    [OSCAR] then claims ~30% better by adding a native OR primitive. So the
    parsed OSCAR ADD should use fewer than 22 NOR and at least one OR --
    numbers from a C++ header checked against two published papers, which is
    about as independent a check as we can get here."""
    add = table["ADD"]
    assert add.mix["NOR"] < 22, "OSCAR ADD should need fewer NOR than MAGIC's 22/bit"
    assert add.mix.get("OR", 0) >= 1, "OSCAR ADD should use the native OR primitive"
    assert add.cycles == 21


def test_cycles_are_positive_and_ordered(table):
    # basic sanity: costs positive, complex ops cost more than simple ones
    for name, rec in table.items():
        assert rec.cycles > 0, f"{name} has no micro-ops"
        assert rec.switching_uops <= rec.cycles

    assert table["MOV"].cycles < table["ADD"].cycles < table["MUL8"].cycles
    assert table["MUL8"].cycles < table["MUL16"].cycles < table["MUL32"].cycles


def test_xor_is_cheaper_than_add(table):
    # no carry chain on a single-bit Boolean primitive -- part of why bitwise
    # kernels are such a good fit for digital PUM
    assert table["XOR"].cycles < table["ADD"].cycles


def test_control_uops_excluded_from_switching_count(table):
    """Energy should only come from micro-ops that actually drive current --
    LOCK/RELEASE/NOP are just pipeline control, shouldn't be charged."""
    mov = table["MOV"]           # LOCK, NOP, NOP, COPY, RELEASE
    assert mov.cycles == 5
    assert mov.switching_uops == 1   # just the COPY


def test_parser_matches_cache(table):
    """Re-parsing the header should agree with the committed snapshot -- keeps
    results reproducible for anyone cloning without running setup.sh."""
    try:
        fresh = recipes.load(refresh=True)
    except FileNotFoundError:
        pytest.skip("MASTODON not cloned; cached snapshot in use")
    assert {k: v.uops for k, v in fresh.items()} == {k: v.uops for k, v in table.items()}
