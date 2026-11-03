"""Digital Compute Element: RACER bit-pipelined Boolean PUM.

Layout first, since it's easy to get backwards. A DCE is 64 independent
bit-pipelines, each 64 arrays deep, arrays are 64x64. Inside one pipeline
([RACER] Sec. 4, [DARTH] Fig. 5): array i holds bit position i of every value
(values get bit-striped down the pipeline, so depth == max operand width, 64
bits here); the 64 rows of an array are 64 vector elements, so a pipeline
works on a 64-wide vector at once; the 64 columns are 64 separate vector
registers. So one pipeline = 64 registers x 64 elements x up to 64 bits, times
64 pipelines per DCE.

Timing: every array can run a different micro-op each cycle -- that's what
makes it bit-*pipelined*, not bit-serial. For a recipe that's R cycles long
(see recipes.py for where R comes from) on a W-bit operand:

  initiation interval = R        (array 0 frees up after R cycles)
  latency = W*R + scan-chain fill (bit i+1 needs bit i's carry, so it's serial)

This distinction actually matters a lot -- a workload that's issuing lots of
independent ops cares about II, not latency, and that's basically the whole
throughput story of this architecture. Mix the two up and you'll understate
digital PUM by ~64x (the operand width) without realizing it.

Energy comes from Table 3 power x active time, same as the paper does it.
switching_energy_pJ is a separate bottom-up estimate from device physics
([RACER] Table 1: 0.0128 pJ/transition), kept around as a sanity check -- the
two numbers should land in the same ballpark, not match exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import recipes
from .area import POWER_MW
from .device import DEVICE, ChipParams, DCEParams


@dataclass(frozen=True)
class OpCost:
    """Cost of one digital vector operation."""

    op: str
    width_bits: int
    elements: int
    latency_cycles: int
    initiation_interval: int
    energy_pJ: float

    @property
    def latency_ns(self) -> float:
        return self.latency_cycles / ChipParams().clock_GHz


class DigitalComputeElement:
    """Timing and energy model for one DCE."""

    def __init__(
        self,
        params: DCEParams | None = None,
        chip: ChipParams | None = None,
    ):
        self.p = params or DCEParams()
        self.chip = chip or ChipParams()
        self.recipes = recipes.load()

    # capacity

    @property
    def elements_per_pipeline(self) -> int:
        """= array rows: that's how many vector elements one pipeline handles at once."""
        return self.p.array.rows

    @property
    def max_elements(self) -> int:
        return self.p.num_pipelines * self.elements_per_pipeline

    @property
    def max_width_bits(self) -> int:
        """one bit per array down the pipeline, so depth caps the operand width"""
        return self.p.pipeline_depth

    # cost model

    def op_cost(self, op: str, width_bits: int = 8, elements: int = 64) -> OpCost:
        """Cost of applying `op` to `elements` values of `width_bits` each.
        Anything over what the DCE holds at once just queues into another
        batch, multiplying the latency."""
        if width_bits > self.max_width_bits:
            raise ValueError(
                f"{width_bits}-bit operand exceeds pipeline depth "
                f"{self.max_width_bits}. Wider values need multiple pipelines "
                "and explicit carry handling."
            )

        rec = self.recipes[op]
        batches = -(-elements // self.max_elements)  # ceil

        latency = batches * (width_bits * rec.cycles + self.p.scan_chain_cycles)
        ii = rec.cycles

        pipelines_busy = min(
            self.p.num_pipelines,
            -(-elements // self.elements_per_pipeline),
        )
        energy = self._energy_pJ(latency, pipelines_busy)

        return OpCost(op, width_bits, elements, latency, ii, energy)

    def _energy_pJ(self, cycles: int, pipelines_busy: int) -> float:
        """Table 3 power x active time. 'Array (Bool Ops)' and 'Pipeline Ctrl'
        are per-pipeline numbers, so only pipelines that actually hold data get
        charged for them."""
        ns = cycles / self.chip.clock_GHz
        mW = (POWER_MW["Array (Bool Ops)"] + POWER_MW["Pipeline Ctrl"]) * pipelines_busy
        return mW * 1e-3 * ns * 1e3  # mW*ns = nJ*1e-3 -> pJ

    def switching_energy_pJ(
        self, op: str, width_bits: int = 8, elements: int = 64
    ) -> float:
        """Bottom-up cross-check on _energy_pJ: count only the micro-ops that
        actually drive current (Boolean/transfer, not control) and charge
        device switching energy for each cell touched.

        Comes out ~20x below the Table 3 number, which is expected, not a bug
        -- this only counts dynamic switching, while Table 3 is whole-element
        active power. The gap is mostly the pipeline control logic, 74000 um^2
        per DCE ([DARTH] Table 3, the single biggest line item), and [RACER]
        Table 4 shows that logic is nearly half static leakage (677 uW static
        vs 775 uW dynamic) that burns regardless of switching. So dynamic
        switching alone badly undercounts the real cost, which is why Table 3
        is what we actually report everywhere.
        """
        rec = self.recipes[op]
        return (rec.switching_uops * width_bits * elements
                * DEVICE.switch_energy_pJ)

    # convenience wrappers

    def sequence_cost(
        self, ops: list[tuple[str, int, int]]
    ) -> tuple[int, float]:
        """Cost of a dependent chain of ops -> (cycles, pJ). Latencies just add
        since nothing overlaps; for independent ops use the II instead."""
        cycles = 0
        energy = 0.0
        for op, width, elements in ops:
            c = self.op_cost(op, width, elements)
            cycles += c.latency_cycles
            energy += c.energy_pJ
        return cycles, energy

    def supports(self, op: str) -> bool:
        return op in self.recipes


if __name__ == "__main__":
    dce = DigitalComputeElement()
    print(f"DCE: {dce.p.num_pipelines} pipelines x {dce.p.pipeline_depth} deep, "
          f"{dce.p.array.rows}x{dce.p.array.cols} arrays")
    print(f"     {dce.max_elements} elements in flight, "
          f"max {dce.max_width_bits}-bit operands\n")

    print(f"{'op':<10}{'width':>6}{'elems':>7}{'latency':>9}{'II':>5}"
          f"{'energy pJ':>12}{'switch pJ':>12}")
    for op in ("XOR", "ADD", "AND", "MUL8", "RELUv1"):
        for width, elems in ((8, 64), (8, 4096), (32, 64)):
            if op == "MUL8" and width != 8:
                continue
            c = dce.op_cost(op, width, elems)
            s = dce.switching_energy_pJ(op, width, elems)
            print(f"{op:<10}{width:>6}{elems:>7}{c.latency_cycles:>9}"
                  f"{c.initiation_interval:>5}{c.energy_pJ:>12.1f}{s:>12.1f}")
