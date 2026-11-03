"""Digital compute element: bit-pipelined timing and energy."""

import pytest

from darthpum.dce import DigitalComputeElement


@pytest.fixture
def dce():
    return DigitalComputeElement()


def test_capacity(dce):
    # 64 pipelines x 64 rows of elements, operand width capped by pipeline depth
    assert dce.elements_per_pipeline == 64
    assert dce.max_elements == 4096
    assert dce.max_width_bits == 64


def test_latency_scales_with_operand_width(dce):
    """Bit i+1 needs bit i's carry, so latency grows linearly with width."""
    narrow = dce.op_cost("ADD", width_bits=8, elements=64).latency_cycles
    wide = dce.op_cost("ADD", width_bits=32, elements=64).latency_cycles
    assert wide > narrow
    assert (narrow - 8) * 4 == (wide - 8)  # 8 cycles of scan-chain fill each time


def test_initiation_interval_is_independent_of_width(dce):
    """This is the whole point of bit-pipelining -- array 0 is free again after
    one recipe length, so a new op can start every R cycles no matter how wide
    the operand is. Mix this up with latency and you'd understate digital PUM
    by a factor of 64 (the operand width), so it's worth pinning down."""
    for w in (8, 16, 32, 64):
        assert dce.op_cost("ADD", width_bits=w, elements=64).initiation_interval == 21


def test_initiation_interval_matches_the_oscar_recipe(dce):
    # II should just be the recipe length, straight from the authors' table
    assert dce.op_cost("XOR").initiation_interval == 9
    assert dce.op_cost("AND").initiation_interval == 7
    assert dce.op_cost("MUL8").initiation_interval == 159


def test_elements_within_capacity_are_free(dce):
    """Up to 4096 elements all run in parallel, so latency should be flat --
    this is where digital PUM's throughput actually comes from."""
    small = dce.op_cost("XOR", 8, 64).latency_cycles
    full = dce.op_cost("XOR", 8, 4096).latency_cycles
    assert small == full


def test_elements_beyond_capacity_batch(dce):
    # past 4096 elements, work just splits into another batch
    one = dce.op_cost("XOR", 8, 4096).latency_cycles
    two = dce.op_cost("XOR", 8, 4097).latency_cycles
    assert two == 2 * one


def test_operand_wider_than_pipeline_is_rejected(dce):
    """128 bits can't be bit-striped across a 64-deep pipeline -- should fail
    loudly here rather than silently truncate and corrupt something later."""
    with pytest.raises(ValueError, match="exceeds pipeline depth"):
        dce.op_cost("ADD", width_bits=128)


def test_unknown_op_is_rejected(dce):
    with pytest.raises(KeyError):
        dce.op_cost("FOURIER_TRANSFORM")


# energy

def test_energy_scales_with_pipelines_actually_used(dce):
    # idle pipelines shouldn't get charged, only ones holding data
    one_pipe = dce.op_cost("XOR", 8, 64).energy_pJ
    all_pipes = dce.op_cost("XOR", 8, 4096).energy_pJ
    assert all_pipes == pytest.approx(64 * one_pipe)


def test_switching_energy_scales_with_element_count(dce):
    a = dce.switching_energy_pJ("XOR", 8, 64)
    b = dce.switching_energy_pJ("XOR", 8, 640)
    assert b == pytest.approx(10 * a)


OPS = ("XOR", "ADD", "AND", "MUL8", "RELUv1")


def test_table3_power_exceeds_bare_switching_energy(dce):
    """Two independent estimates of the same thing: Table 3 active power x
    time vs. device switching energy from the recipe. Table 3 has to come out
    higher since it includes the pipeline control logic's static power --
    74000 um^2/DCE, the biggest single component, and [RACER] Table 4 says
    it's nearly half leakage (677 uW static vs 775 uW dynamic) regardless of
    whether anything's actually switching."""
    for op in OPS:
        ratio = (dce.op_cost(op, 8, 4096).energy_pJ
                 / dce.switching_energy_pJ(op, 8, 4096))
        assert 5 < ratio < 50, f"{op}: implausible ratio {ratio:.1f}"


def test_energy_ratio_is_invariant_to_element_count(dce):
    """Both models scale linearly with element count, so their ratio shouldn't
    budge just because we change how many elements there are."""
    for op in OPS:
        r64 = dce.op_cost(op, 8, 64).energy_pJ / dce.switching_energy_pJ(op, 8, 64)
        r4k = dce.op_cost(op, 8, 4096).energy_pJ / dce.switching_energy_pJ(op, 8, 4096)
        assert r64 == pytest.approx(r4k), f"{op}: ratio moved with element count"


def test_energy_ratio_tracks_idle_cycles_per_operation(dce):
    """The ratio should differ between ops, but predictably: an op whose
    recipe spends more cycles on control micro-ops burns Table 3 power
    without switching anything, so it gets a higher ratio. AND is 7 cycles / 4
    switching (~23); ADD is 21 cycles / 19 switching (~13.6). Normalize out
    cycles/switching_uops and the spread should mostly disappear -- that's
    the evidence the two models agree structurally and aren't just
    coincidentally close."""
    normalised = []
    for op in OPS:
        rec = dce.recipes[op]
        ratio = (dce.op_cost(op, 8, 4096).energy_pJ
                 / dce.switching_energy_pJ(op, 8, 4096))
        normalised.append(ratio / (rec.cycles / rec.switching_uops))

    spread = max(normalised) / min(normalised)
    assert spread < 1.2, f"normalised ratios still diverge: {normalised}"


def test_dependent_sequence_latencies_add(dce):
    # no overlap on a dependent chain, so cost is just the sum of latencies
    cycles, energy = dce.sequence_cost([("XOR", 8, 64), ("ADD", 8, 64)])
    assert cycles == (dce.op_cost("XOR", 8, 64).latency_cycles
                      + dce.op_cost("ADD", 8, 64).latency_cycles)
    assert energy > 0


def test_supports_reports_available_operations(dce):
    assert dce.supports("ADD")
    assert not dce.supports("SQUARE_ROOT")
