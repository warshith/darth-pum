"""Hybrid Compute Tile: the coordination layer between analog and digital PUM.

This is the part of DARTH-PUM nobody else models. MASTODON does digital PUM,
CrossSim does analog crossbars, and neither one knows the other exists -- the
glue connecting them is basically the paper's whole contribution, and this file
is our attempt at reconstructing it.

Why it's not just "wire them together": analog and digital PUM don't compose
for free, three separate reasons.

1. Perpendicular axes -- the analog array takes a row-wise input and sums down
   bitlines, so you get a *row* vector out. Digital PUM stripes data
   column-wise and works along rows instead. So anything crossing the
   boundary needs a transpose ([DARTH] Sec. 4.2).

2. Latency mismatch -- an analog MVM is hundreds of cycles (mostly conversion);
   a digital Boolean op is a handful. Let them run free and a younger digital
   instruction could race ahead of an older analog one it actually depends on.
   Hence the arbiter, which just serializes them.

3. Partial products -- bit-slicing an N-bit input gives you N partial products,
   each needing its own shift before it can be added in. Shifts, writes into
   the digital arrays, and the adds all fight over the same pipelines.

#3 is the expensive one, so that's where the paper's actual optimization
lives (Figures 9 and 10). Naively (Fig. 10a) you write a bit-slice in (N
cycles), shift it into place (M cycles for depth M), then add -- and RACER's
bit-pipelined model can't interleave those steps without wrecking the
pipeline, so they just run back to back. Analog sits there idle the whole
time holding a finished result with nowhere to put it.

DARTH-PUM's fix: shift units that do the shifting *during* the transfer
instead of after. The shift amount is known ahead of time from the bit
position, so there's no reconfigurable routing needed, nothing dynamic. Per
the paper this buys three things: no shift warm-up/cool-down per partial
product, ADC output rate can be matched to digital write bandwidth, and the
adds become pipelineable instead of serial.

Both timelines (10a and 10b) are implemented below so the gap between them
is a real, measured number, not an assumption -- it tells you how much of
the whole architecture's benefit is riding on this one trick.
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
    stall_cycles: int  # time the arbiter spent blocking one side on the other

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
        self.optimized = optimized  # False = the Figure 10a (unoptimized) timeline

    # -- the ACE<->DCE interconnect --

    def transfer_cycles(self, values: int, bits_per_value: int) -> int:
        """Move a partial-product vector from the ACE to the DCE. The on-chip
        network is 8 B/cycle -- [DARTH] Sec. 4 picks that width specifically to
        rate-match ADC throughput against DCE write bandwidth."""
        total_bytes = math.ceil(values * bits_per_value / 8)
        return math.ceil(total_bytes / self.chip.io_bytes_per_cycle)

    def shift_cycles(self, bit_position: int) -> int:
        """Cost of placing a partial product at its bit position. Free when
        optimized (the shift units do it during the transfer, the shift amount
        is already known from the bit position). Otherwise the data has to be
        written in and then walked up the pipeline one position per cycle --
        the slow part of Figure 10a."""
        return 0 if self.optimized else bit_position

    # the main operation

    def mvm(
        self,
        rows: int,
        cols: int,
        input_bits: int = 8,
        output_bits: int | None = None,
        value_bits: int = 8,
        active_row_fraction: float = 1.0,
    ) -> HCTCost:
        """Full bit-sliced MVM across both domains: analog produces one
        partial product per input bit-slice, digital shifts each into place
        and accumulates. active_row_fraction just passes through to
        AnalogComputeElement.mvm_cost (the zero-skip contribution) -- leave it
        at 1.0 and this is identical to every other call site."""
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
        """Figure 10b: shift happens during the transfer, adds pipeline behind
        the analog work. Analog streams partial products back to back, each
        transfer's shift is free, adds pipeline at the II -- so steady state is
        bound by whichever of the three stages is slowest, and only the final
        add's latency shows up at the end."""
        stage = max(analog_per_slice, xfer, add_ii)
        cycles = slices * stage + add_latency
        stall = max(0, (stage - analog_per_slice) * slices)
        return cycles, stall

    def _unoptimized(
        self, slices: int, analog_per_slice: int, xfer: int,
        add_latency: int, elements: int,
    ) -> tuple[int, int]:
        """Figure 10a: write, then shift, then add. Fully serial. [DARTH] Sec
        4.1 spells it out: "for a pipeline with N elements per vector register
        and a depth of M, it would require N cycles to write followed by M
        cycles to shift the data before the next set of partial products can
        be written." Analog just stalls the whole time, since there's nowhere
        to put its next partial product."""
        write = elements                      # one row of data per cycle
        total = 0
        stall = 0
        for bit in range(slices):
            serial = write + self.shift_cycles(bit) + add_latency
            total += analog_per_slice + xfer + serial
            stall += serial                   # analog idle for all of it
        return total, stall

    def _frontend_energy_pJ(self, cycles: int) -> float:
        """Fetch/decode/issue cost, split across the tiles sharing one front end."""
        ns = cycles / self.chip.clock_GHz
        return (FRONTEND_POWER_MW / self.chip.hcts_per_frontend) * ns

    # pure-digital path, no analog involved at all

    def digital_op(self, op: str, width_bits: int = 8,
                   elements: int = 64) -> HCTCost:
        """Ops with no analog component -- ReLU, XOR, softmax, that kind of thing."""
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
