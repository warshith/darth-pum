"""Digital Compute Element: RACER bit-pipelined Boolean PUM.

Data layout
-----------
A DCE holds 64 independent bit-pipelines, each 64 arrays deep, every array 64x64.
Within one pipeline ([RACER] Section 4, [DARTH] Figure 5):

  * array i holds **bit position i** of every value -- values are bit-striped
    across the depth of the pipeline, so pipeline depth is the maximum operand
    width (64 bits here);
  * the 64 **rows** of an array are 64 different vector elements, so one pipeline
    operates on a 64-element vector at once;
  * the 64 **columns** are 64 distinct vector registers.

So one pipeline stores 64 vector registers x 64 elements x up to 64 bits, and a
DCE has 64 such pipelines.

Timing
------
Each array can execute a different micro-op each cycle, which is what makes the
design bit-*pipelined* rather than merely bit-serial. For an op whose recipe is
R cycles long (from the OSCAR recipe book -- see recipes.py) applied to a W-bit
operand:

  * **initiation interval = R.** Array 0 is free again after R cycles, so a new
    operation can start every R cycles.
  * **latency = W * R + scan-chain fill.** Bit i+1 depends on bit i (carry
    propagation), so the bit positions proceed in sequence down the pipeline.

The distinction matters: a workload issuing many independent vector operations is
limited by the initiation interval, not the latency, and that is precisely the
throughput advantage the architecture is built around. Code that conflates the two
will understate digital PUM by roughly the operand width -- a factor of 64 here.

Energy
------
Charged from the Table 3 power figures multiplied by active time, which is how the
paper's own numbers are expressed. `switching_energy_pJ` independently estimates
the same quantity from device physics ([RACER] Table 1, 0.0128 pJ per transition)
as a cross-check; the two should agree to order of magnitude.
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

    # -- capacity -------------------------------------------------------

    @property
    def elements_per_pipeline(self) -> int:
        """Vector elements processed in parallel by one pipeline = array rows."""
        return self.p.array.rows

    @property
    def max_elements(self) -> int:
        """Elements in flight across the whole DCE."""
        return self.p.num_pipelines * self.elements_per_pipeline

    @property
    def max_width_bits(self) -> int:
        """Widest operand: one bit per array down the pipeline."""
        return self.p.pipeline_depth

    # -- cost -----------------------------------------------------------

    def op_cost(self, op: str, width_bits: int = 8, elements: int = 64) -> OpCost:
        """Cost of applying `op` to `elements` values of `width_bits` each.

        Elements beyond what the DCE holds at once are processed in successive
        batches, which multiplies the latency.
        """
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
        """Table 3 power x active time.

        'Array (Bool Ops)' and 'Pipeline Ctrl' are per-pipeline figures; only the
        pipelines actually holding data are charged.
        """
        ns = cycles / self.chip.clock_GHz
        mW = (POWER_MW["Array (Bool Ops)"] + POWER_MW["Pipeline Ctrl"]) * pipelines_busy
        return mW * 1e-3 * ns * 1e3  # mW*ns = nJ*1e-3 -> pJ

    def switching_energy_pJ(
        self, op: str, width_bits: int = 8, elements: int = 64
    ) -> float:
        """Independent bottom-up energy estimate, as a cross-check on _energy_pJ.

        Counts micro-ops that actually drive current (Boolean and transfer ops,
        not pipeline control) and charges each cell they switch, across every
        element touched.

        This lands roughly 20x *below* the Table 3 estimate, and the gap is
        expected rather than an error: it counts only dynamic switching, whereas
        the Table 3 figures are whole-element active power. Most of that
        difference is the bit-pipeline control logic, which [DARTH] Table 3 puts
        at 74000 um^2 per DCE -- by far the largest component -- and which
        [RACER] Table 4 measures at 677 uW static against 775 uW dynamic, i.e.
        nearly half its power is leakage that burns whether or not devices
        switch. Dynamic switching energy alone therefore understates the real
        cost badly, which is why the Table 3 figures are what we report.
        """
        rec = self.recipes[op]
        return (rec.switching_uops * width_bits * elements
                * DEVICE.switch_energy_pJ)

    # -- convenience ----------------------------------------------------

    def sequence_cost(
        self, ops: list[tuple[str, int, int]]
    ) -> tuple[int, float]:
        """Cost of a dependent chain of operations: (cycles, pJ).

        Dependent, so latencies add -- no overlap. Independent work should be
        costed with the initiation interval instead.
        """
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
