"""Digital compute element: bit-pipelined timing and energy."""

import pytest

from darthpum.dce import DigitalComputeElement


@pytest.fixture
def dce():
    return DigitalComputeElement()


def test_capacity(dce):
    """64 pipelines x 64 rows of elements; operand width capped by pipeline depth."""
    assert dce.elements_per_pipeline == 64
    assert dce.max_elements == 4096
    assert dce.max_width_bits == 64


def test_latency_scales_with_operand_width(dce):
    """Bit i+1 waits on bit i's carry, so the bit positions proceed in sequence
    and latency grows linearly with width."""
    narrow = dce.op_cost("ADD", width_bits=8, elements=64).latency_cycles
    wide = dce.op_cost("ADD", width_bits=32, elements=64).latency_cycles
    assert wide > narrow
    # 8 extra cycles of scan-chain fill at each width.
    assert (narrow - 8) * 4 == (wide - 8)


def test_initiation_interval_is_independent_of_width(dce):
    """The point of bit-pipelining: array 0 frees up after one recipe length, so a
    new operation can start every R cycles however wide the operand is.

    Conflating this with latency would understate digital PUM by the operand
    width -- a factor of 64 here -- so it is worth pinning explicitly.
    """
    for w in (8, 16, 32, 64):
        assert dce.op_cost("ADD", width_bits=w, elements=64).initiation_interval == 21


def test_initiation_interval_matches_the_oscar_recipe(dce):
    """II must be the recipe length straight from the authors' cost table."""
    assert dce.op_cost("XOR").initiation_interval == 9
    assert dce.op_cost("AND").initiation_interval == 7
    assert dce.op_cost("MUL8").initiation_interval == 159


def test_elements_within_capacity_are_free(dce):
    """Up to 4096 elements run in parallel, so latency is flat across that range.
    This parallelism is where digital PUM's throughput comes from."""
    small = dce.op_cost("XOR", 8, 64).latency_cycles
    full = dce.op_cost("XOR", 8, 4096).latency_cycles
    assert small == full


def test_elements_beyond_capacity_batch(dce):
    """Past 4096 elements the work splits into successive batches."""
    one = dce.op_cost("XOR", 8, 4096).latency_cycles
    two = dce.op_cost("XOR", 8, 4097).latency_cycles
    assert two == 2 * one


def test_operand_wider_than_pipeline_is_rejected(dce):
    """A 128-bit value cannot be bit-striped across a 64-deep pipeline. Failing
    loudly beats silently truncating, which would corrupt results downstream."""
    with pytest.raises(ValueError, match="exceeds pipeline depth"):
        dce.op_cost("ADD", width_bits=128)


def test_unknown_op_is_rejected(dce):
    with pytest.raises(KeyError):
        dce.op_cost("FOURIER_TRANSFORM")


# -- energy -------------------------------------------------------------

def test_energy_scales_with_pipelines_actually_used(dce):
    """Only pipelines holding data are charged; idle ones are not."""
    one_pipe = dce.op_cost("XOR", 8, 64).energy_pJ
    all_pipes = dce.op_cost("XOR", 8, 4096).energy_pJ
    assert all_pipes == pytest.approx(64 * one_pipe)


def test_switching_energy_scales_with_element_count(dce):
    """Bottom-up device energy must track the number of cells switched."""
    a = dce.switching_energy_pJ("XOR", 8, 64)
    b = dce.switching_energy_pJ("XOR", 8, 640)
    assert b == pytest.approx(10 * a)


OPS = ("XOR", "ADD", "AND", "MUL8", "RELUv1")


def test_table3_power_exceeds_bare_switching_energy(dce):
    """The two energy models are independent estimates of the same quantity:
    Table 3 active power x time, versus device switching energy counted from the
    recipe. Table 3 must come out higher, because it includes the static power of
    the pipeline control logic -- 74000 um^2 per DCE, the largest single
    component, which [RACER] Table 4 measures at 677 uW static against 775 uW
    dynamic. Nearly half of it leaks whether or not devices switch.
    """
    for op in OPS:
        ratio = (dce.op_cost(op, 8, 4096).energy_pJ
                 / dce.switching_energy_pJ(op, 8, 4096))
        assert 5 < ratio < 50, f"{op}: implausible ratio {ratio:.1f}"


def test_energy_ratio_is_invariant_to_element_count(dce):
    """Both models scale linearly in the number of elements, so their ratio must
    not move at all when only the element count changes."""
    for op in OPS:
        r64 = dce.op_cost(op, 8, 64).energy_pJ / dce.switching_energy_pJ(op, 8, 64)
        r4k = dce.op_cost(op, 8, 4096).energy_pJ / dce.switching_energy_pJ(op, 8, 4096)
        assert r64 == pytest.approx(r4k), f"{op}: ratio moved with element count"


def test_energy_ratio_tracks_idle_cycles_per_operation(dce):
    """The ratio *should* differ between operations, and in a predictable way.

    An operation whose recipe spends many cycles on control micro-ops burns Table
    3 power without switching devices, so its ratio is higher. AND is 7 cycles
    with only 4 switching (ratio ~23); ADD is 21 cycles with 19 switching (~13.6).
    Dividing out cycles/switching_uops should therefore collapse the spread --
    which is the evidence the two models are structurally consistent rather than
    coincidentally close.
    """
    normalised = []
    for op in OPS:
        rec = dce.recipes[op]
        ratio = (dce.op_cost(op, 8, 4096).energy_pJ
                 / dce.switching_energy_pJ(op, 8, 4096))
        normalised.append(ratio / (rec.cycles / rec.switching_uops))

    spread = max(normalised) / min(normalised)
    assert spread < 1.2, f"normalised ratios still diverge: {normalised}"


def test_dependent_sequence_latencies_add(dce):
    """A dependent chain cannot overlap, so its cost is the sum of latencies."""
    cycles, energy = dce.sequence_cost([("XOR", 8, 64), ("ADD", 8, 64)])
    assert cycles == (dce.op_cost("XOR", 8, 64).latency_cycles
                      + dce.op_cost("ADD", 8, 64).latency_cycles)
    assert energy > 0


def test_supports_reports_available_operations(dce):
    assert dce.supports("ADD")
    assert not dce.supports("SQUARE_ROOT")
