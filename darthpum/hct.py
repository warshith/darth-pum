"""Hybrid Compute Tile: the coordination layer between analog and digital PUM.

This is the part of DARTH-PUM that exists nowhere else. MASTODON models digital
PUM, CrossSim models analog crossbars, and neither knows about the other -- the
glue between them is the paper's contribution, and this module is our
reconstruction of it.

The problem being solved
------------------------
Analog and digital PUM do not compose naturally. Three mismatches:

1. **Perpendicular axes.** The analog array takes a row-wise (wordline) input and
   accumulates down bitlines, so a result arrives as a *row* vector. Digital PUM
   stripes data column-wise and computes along rows. Data crossing between the
   domains must be transposed ([DARTH] Section 4.2).

2. **Latency mismatch.** An analog MVM takes hundreds of cycles, dominated by
   conversion; a digital Boolean primitive takes a handful. Letting them
   interleave freely would let a younger digital instruction overtake an older
   analog one it depends on, so an arbiter serialises them.

3. **Partial products.** Input bit-slicing means an N-bit input produces N
   separate partial products, each needing a shift to its bit position before it
   can be summed. The shifts, the writes into the digital arrays, and the adds
   all contend for the same pipelines.

The optimisation ([DARTH] Figures 9 and 10)
-------------------------------------------
Mismatch 3 is the expensive one. Naively (Figure 10a), each bit-slice must be
written into the digital pipeline (N cycles for an N-element vector register),
then shifted into position (M cycles for a depth-M pipeline), then added -- and
because RACER's bit-pipelined execution model cannot interleave these without
disrupting the pipeline, they serialise. The analog side meanwhile sits idle
waiting, having already produced the next partial product.

DARTH-PUM's answer is a bank of shift units that shift *during* the transfer from
analog to digital, rather than after it. The shift amount is known in advance from
the bit position, so no software intervention or reconfigurable interconnect is
needed. That buys three things, per the paper: no repeated shift warm-up/cool-down
per partial product, the ADC output rate can be matched to the digital write
bandwidth, and all the adds become pipelineable.

Both timelines are implemented here. The gap between them is an internal
reproduction of Figure 10, and it isolates how much of DARTH-PUM's benefit comes
from this one design decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .ace import AnalogComputeElement, MVMCost
from .area import FRONTEND_POWER_MW
from .dce import DigitalComputeElement
from .device import ACEParams, ChipParams, DCEParams


@dataclass(frozen=True)
class HCTCost:
    """Cost of one end-to-end operation on a hybrid tile."""

    cycles: int
    energy_pJ: float
    analog_cycles: int
    digital_cycles: int
    stall_cycles: int
    """Cycles the arbiter spent blocking one domain on the other."""

    @property
    def latency_ns(self) -> float:
        return self.cycles / ChipParams().clock_GHz


class HybridComputeTile:
    """One HCT: an analog element, a digital element, and the glue between them."""

    def __init__(
        self,
        ace_params: ACEParams | None = None,
        dce_params: DCEParams | None = None,
        chip: ChipParams | None = None,
        optimized: bool = True,
    ):
        self.chip = chip or ChipParams()
        self.ace = AnalogComputeElement(ace_params, self.chip)
        self.dce = DigitalComputeElement(dce_params, self.chip)
        self.optimized = optimized
        """False reproduces the Figure 10a timeline, for the ablation."""

    # -- interconnect ---------------------------------------------------

    def transfer_cycles(self, values: int, bits_per_value: int) -> int:
        """Move a partial-product vector from the ACE to the DCE.

        The on-chip network carries 8 B/cycle, a width [DARTH] Section 4 says was
        chosen to rate-match ADC throughput against DCE write bandwidth.
        """
        total_bytes = math.ceil(values * bits_per_value / 8)
        return math.ceil(total_bytes / self.chip.io_bytes_per_cycle)

    def shift_cycles(self, bit_position: int) -> int:
        """Cost of placing a partial product at its bit position.

        Optimised: free. The shift units act during the transfer, and the shift
        amount is fixed by the bit position, so it costs no extra cycles.

        Unoptimised: the data must be written in, then walked up the pipeline one
        position per cycle -- the blue operations of Figure 10a.
        """
        return 0 if self.optimized else bit_position

    # -- the main operation ---------------------------------------------

    def mvm(
        self,
        rows: int,
        cols: int,
        input_bits: int = 8,
        output_bits: int | None = None,
        value_bits: int = 8,
        active_row_fraction: float = 1.0,
    ) -> HCTCost:
        """A full bit-sliced matrix-vector multiply across both domains.

        The analog side produces one partial product per input bit-slice; the
        digital side shifts each into position and accumulates them.

        `active_row_fraction`: forwarded to AnalogComputeElement.mvm_cost --
        the zero-skipping contribution. 1.0 (default) leaves this call
        identical to every other workload's.
        """
        mvm_cost = self.ace.mvm_cost(rows, cols, input_bits, output_bits, value_bits,
                                     active_row_fraction=active_row_fraction)
        out_bits = mvm_cost.output_bits
        elements = min(cols, self.ace.p.array.cols)

        # Per-slice analog cost: the whole MVM is input_bits slices.
        analog_per_slice = mvm_cost.cycles // input_bits

        xfer = self.transfer_cycles(elements, out_bits)
        add = self.dce.op_cost("ADD", width_bits=out_bits + input_bits,
                               elements=elements)

        if self.optimized:
            cycles, stall = self._optimized(
                input_bits, analog_per_slice, xfer, add.latency_cycles,
                add.initiation_interval)
        else:
            cycles, stall = self._unoptimized(
                input_bits, analog_per_slice, xfer, add.latency_cycles, elements)

        digital_cycles = cycles - mvm_cost.cycles + stall
        energy = (mvm_cost.energy_pJ
                  + add.energy_pJ * input_bits
                  + self._frontend_energy_pJ(cycles))

        return HCTCost(
            cycles=cycles,
            energy_pJ=energy,
            analog_cycles=mvm_cost.cycles,
            digital_cycles=max(digital_cycles, 0),
            stall_cycles=stall,
        )

    def _optimized(
        self, slices: int, analog_per_slice: int, xfer: int,
        add_latency: int, add_ii: int,
    ) -> tuple[int, int]:
        """Figure 10b: shift during transfer, adds pipelined behind the analog work.

        The analog element streams partial products back to back. Each transfer
        carries its shift for free, and the adds pipeline at the initiation
        interval. So the steady state is limited by whichever of the three stages
        is slowest, and only the final add's latency is exposed at the end.
        """
        stage = max(analog_per_slice, xfer, add_ii)
        cycles = slices * stage + add_latency
        stall = max(0, (stage - analog_per_slice) * slices)
        return cycles, stall

    def _unoptimized(
        self, slices: int, analog_per_slice: int, xfer: int,
        add_latency: int, elements: int,
    ) -> tuple[int, int]:
        """Figure 10a: write, then shift, then add -- strictly serialised.

        [DARTH] Section 4.1: "for a pipeline with N elements per vector register
        and a depth of M, it would require N cycles to write followed by M cycles
        to shift the data before the next set of partial products can be written".
        The analog element stalls throughout, because its next partial product has
        nowhere to go.
        """
        write = elements                      # one row of data per cycle
        total = 0
        stall = 0
        for bit in range(slices):
            serial = write + self.shift_cycles(bit) + add_latency
            total += analog_per_slice + xfer + serial
            stall += serial                   # analog idle for all of it
        return total, stall

    def _frontend_energy_pJ(self, cycles: int) -> float:
        """Fetch/decode/issue, amortised over the tiles sharing one front end."""
        ns = cycles / self.chip.clock_GHz
        return (FRONTEND_POWER_MW / self.chip.hcts_per_frontend) * ns

    # -- digital-only path ----------------------------------------------

    def digital_op(self, op: str, width_bits: int = 8,
                   elements: int = 64) -> HCTCost:
        """An operation with no analog component (ReLU, XOR, softmax, ...)."""
        c = self.dce.op_cost(op, width_bits, elements)
        return HCTCost(
            cycles=c.latency_cycles,
            energy_pJ=c.energy_pJ + self._frontend_energy_pJ(c.latency_cycles),
            analog_cycles=0,
            digital_cycles=c.latency_cycles,
            stall_cycles=0,
        )


if __name__ == "__main__":
    print("Figure 10 ablation: what the shift-during-transfer units buy.\n")
    print(f"{'matrix':<12}{'in bits':>8}{'Fig 10a':>10}{'Fig 10b':>10}"
          f"{'speedup':>9}{'10a stall':>11}")

    for rows, cols, ib in ((64, 64, 8), (64, 64, 4), (576, 64, 8), (288, 32, 8)):
        a = HybridComputeTile(optimized=False).mvm(rows, cols, input_bits=ib,
                                                   value_bits=1)
        b = HybridComputeTile(optimized=True).mvm(rows, cols, input_bits=ib,
                                                  value_bits=1)
        print(f"{f'{rows}x{cols}':<12}{ib:>8}{a.cycles:>10}{b.cycles:>10}"
              f"{a.cycles / b.cycles:>8.2f}x{a.stall_cycles:>11}")

    print("\nThe stall column is cycles the analog element spends idle waiting on")
    print("the digital side -- which is exactly what the shift units remove.")
