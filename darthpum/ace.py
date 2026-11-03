"""Analog Compute Element: ReRAM crossbar matrix-vector multiply.

Matrix values live as device conductances. Drive wordline i with voltage V_i
and each device passes I = V_i * G_ij (Ohm's law); every device on bitline j
dumps its current onto that wire (Kirchhoff). So the current on bitline j is
just sum_i V_i * G_ij -- a dot product for free, all 64 bitlines at once, no
clocked multiply-accumulate at all. That's the whole appeal of doing this in
analog.

Of course the catch is at the edges, not the middle. Devices only hold a
couple of bits reliably and inputs/outputs have to be converted to/from
voltages, so:

  * matrix bit-slicing -- a W-bit weight gets split across ceil(W/cell_bits)
    arrays, partial products recombined by shift-and-add ([DARTH] Fig. 2). SLC
    (1 bit/cell) means an 8-bit weight eats 8 arrays.
  * input bit-slicing -- instead of a high-resolution DAC, an N-bit input goes
    in one bit per cycle over N cycles ([DARTH] Sec. 2.2.1). Side benefit: no
    DAC non-linearity to worry about.

Digitizing the bitlines is where most of the cost actually lives, and the two
ADC families fall apart in opposite ways:

  * SAR -- binary search, fast, but one bitline at a time, so a couple of ADCs
    get time-multiplexed across all 64. Cost ~ ceil(64/n_adc) cycles,
    independent of how many output bits you actually want.
  * Ramp -- sweeps a reference and compares. Slow in the worst case (256
    cycles at full 8-bit res) but it hits every bitline at once, and you can
    bail out early once you have enough output bits. Cost ~ 2^out_bits cycles.

Net result: which converter wins depends on the workload, not the chip. A
kernel that only needs 2 output bits should clearly use ramp; ResNet-20 wants
~8 bits, where ramp's full 256 cycles loses badly to SAR's 32 (an 8x gap,
roughly consistent with the paper's overall 1.5x SAR advantage once you
average across its whole workload mix -- [DARTH] Sec. 7.3).

The paper just commits to one converter chip-wide and eats the loss on
whichever workload wanted the other one. We model both so that loss is
actually measurable instead of hand-waved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .area import POWER_MW
from .device import ACEParams, ChipParams


@dataclass(frozen=True)
class MVMCost:
    """Cost of one matrix-vector multiply on the analog arrays."""

    rows: int
    cols: int
    input_bits: int
    output_bits: int
    arrays_used: int
    cycles: int
    energy_pJ: float
    adc_cycles: int  # conversion alone, usually most of `cycles`

    @property
    def adc_fraction(self) -> float:
        return self.adc_cycles / self.cycles if self.cycles else 0.0

    @property
    def latency_ns(self) -> float:
        return self.cycles / ChipParams().clock_GHz


class AnalogComputeElement:
    """Timing, energy and functional model for one ACE."""

    def __init__(
        self,
        params: ACEParams | None = None,
        chip: ChipParams | None = None,
    ):
        self.p = params or ACEParams()
        self.chip = chip or ChipParams()

    # mapping

    def tiles_for_matrix(self, rows: int, cols: int, value_bits: int = 8) -> int:
        """Arrays needed to hold a rows x cols matrix (the ISAAC/PUMA tiling:
        chop into array-sized blocks, zero-pad the leftovers, replicate across
        bit-slices). Row-tiles still need a digital accumulate afterward."""
        slices = math.ceil(value_bits / self.p.cell_bits)
        return (math.ceil(rows / self.p.array.rows)
                * math.ceil(cols / self.p.array.cols)
                * slices)

    def fits(self, rows: int, cols: int, value_bits: int = 8) -> bool:
        return self.tiles_for_matrix(rows, cols, value_bits) <= self.p.num_arrays

    # -- ADC / conversion cost --

    def adc_cycles_per_slice(self, output_bits: int | None = None) -> int:
        """Cycles to digitise one array's bitlines after a single input bit-slice.

        This is the one function that encodes the whole SAR-vs-ramp story, so
        it's also what our contribution hooks into.
        """
        adc = self.p.adc
        bits = output_bits if output_bits is not None else adc.resolution_bits

        if adc.kind == "SAR":
            # One bitline at a time; the ADCs are multiplexed across all of them.
            # Latency per conversion is independent of the precision requested.
            return math.ceil(self.p.array.cols / adc.count) * adc.latency_cycles

        if adc.kind == "RAMP":
            # All bitlines at once, but the reference sweeps 2^bits increments.
            # Early termination is what lets a low-output-bit kernel win here.
            full = adc.latency_cycles
            needed = min(2 ** bits, full)
            return math.ceil(needed / adc.count)

        raise ValueError(f"unknown ADC kind {adc.kind!r}")

    # -- the actual cost model --

    def mvm_cost(
        self,
        rows: int,
        cols: int,
        input_bits: int = 8,
        output_bits: int | None = None,
        value_bits: int = 8,
        active_row_fraction: float = 1.0,
    ) -> MVMCost:
        """Cost of multiplying a rows x cols matrix by one input vector.

        Row-tile partial sums get summed digitally afterward by the DCE --
        that's hct.py's job, not this function's.

        active_row_fraction is the zero-skipping contribution: a wordline
        whose input bit is exactly zero drives no current anywhere, so its
        driver can just be switched off for the MVM. This only ever saves
        energy, never cycles -- the bitline is still a sum over whatever rows
        stayed active and the ADC still has to digitize it either way, so
        cycle counts and ADC cost are untouched. All it scales down is the
        "Row Periphery" term in _energy_pJ (the wordline-driver slice of
        Table 3). Default 1.0 means nothing here changes from before this
        contribution existed.

        End-to-end on ResNet-20 (measured sparsity, real checkpoint, real
        CIFAR-10 -- see sparsity.py) this is worth about 3% total energy, not
        the ~48% a naive row-driver-only estimate would suggest. Row
        Periphery is just a small slice of ACE energy once you account for
        ADC conversion (always running, doesn't care about zeros) and
        recombining the bit-sliced partial sums digitally (also doesn't care
        -- up to 72 adds per layer for INT8-on-SLC, every single time).
        """
        output_bits = (output_bits if output_bits is not None
                       else self.p.adc.resolution_bits)

        tiles = self.tiles_for_matrix(rows, cols, value_bits)
        col_tiles = math.ceil(cols / self.p.array.cols)
        slices = math.ceil(value_bits / self.p.cell_bits)

        # Arrays operate in parallel, limited by how many the ACE has.
        waves = math.ceil(tiles / self.p.num_arrays)

        per_slice = 1 + self.adc_cycles_per_slice(output_bits)  # 1 drive cycle + convert
        adc_only = self.adc_cycles_per_slice(output_bits)

        cycles = waves * input_bits * per_slice
        adc_cycles = waves * input_bits * adc_only

        active = min(tiles, self.p.num_arrays)
        energy = self._energy_pJ(cycles, adc_cycles, active, active_row_fraction)

        return MVMCost(
            rows=rows, cols=cols,
            input_bits=input_bits, output_bits=output_bits,
            arrays_used=tiles, cycles=cycles,
            energy_pJ=energy, adc_cycles=adc_cycles,
        )

    def _energy_pJ(self, cycles: int, adc_cycles: int, active_arrays: int,
                  active_row_fraction: float = 1.0) -> float:
        """Table 3 power x active time. Row periphery + S&H burn for the whole
        op; ADCs only while actually converting. active_row_fraction only
        touches Row Periphery (the wordline drivers, the bit zero-skipping
        turns off) -- S&H and the ADCs don't care how many rows were active,
        they still sample/convert the bitline regardless."""
        ns = cycles / self.chip.clock_GHz
        adc_ns = adc_cycles / self.chip.clock_GHz

        array_mW = (POWER_MW["Row Periphery"] * active_row_fraction
                   + POWER_MW["S&H (Analog)"]) * active_arrays
        adc_key = "SAR ADC" if self.p.adc.kind == "SAR" else "Ramp ADC"
        adc_mW = POWER_MW[adc_key] * self.p.adc.count

        return (array_mW * ns + adc_mW * adc_ns)  # mW*ns == pJ

    # functional model (not the timing model -- just "what number comes out")

    def mvm(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
        output_bits: int | None = None,
        ideal: bool = True,
    ) -> np.ndarray:
        """Evaluate matrix @ vector the way the hardware would.

        ideal=True should be exact and match plain numpy -- the test suite
        checks this, so a bit-slicing bug can't hide behind "close enough".
        Non-idealities (noise, drift, etc.) are handled separately via
        CrossSim, kept out of this file on purpose so the timing model and the
        accuracy model don't end up tangled together.
        """
        matrix = np.asarray(matrix)
        vector = np.asarray(vector)

        if matrix.shape[-1] != vector.shape[0]:
            raise ValueError(
                f"shape mismatch: matrix {matrix.shape} @ vector {vector.shape}"
            )

        out = matrix.astype(np.int64) @ vector.astype(np.int64)

        if not ideal and output_bits is not None:
            # Saturating quantisation to the ADC's output range.
            lim = 2 ** output_bits - 1
            out = np.clip(out, 0, lim)

        return out

    def mvm_bitsliced(self, matrix: np.ndarray, vector: np.ndarray,
                      input_bits: int = 8) -> np.ndarray:
        """Same thing, but via explicit input bit-slicing like the real hardware:
        one input bit per cycle, partial products recombined by shifting. Should
        equal the direct product exactly -- if it doesn't, the dataflow model
        is wrong somewhere, not just imprecise."""
        matrix = np.asarray(matrix, dtype=np.int64)
        vector = np.asarray(vector, dtype=np.int64)

        acc = np.zeros(matrix.shape[0], dtype=np.int64)
        for bit in range(input_bits):
            slice_in = (vector >> bit) & 1          # one wordline bit per cycle
            partial = matrix @ slice_in             # physics does this in one shot
            acc += partial << bit                   # shift-and-add in the DCE
        return acc


if __name__ == "__main__":
    from .device import ADCParams

    rng = np.random.default_rng(0)
    m = rng.integers(0, 16, size=(64, 64))
    v = rng.integers(0, 256, size=64)

    ace = AnalogComputeElement()
    exact = m @ v
    sliced = ace.mvm_bitsliced(m, v, input_bits=8)
    print(f"bit-sliced MVM == direct product: {np.array_equal(exact, sliced)}\n")

    print("Cost of one 64x64 MVM, by converter and required output precision:")
    print(f"{'ADC':<6}{'out bits':>9}{'cycles':>9}{'ADC cyc':>9}"
          f"{'ADC %':>8}{'energy pJ':>12}")
    for name, adc in (("SAR", ADCParams.sar()), ("RAMP", ADCParams.ramp())):
        a = AnalogComputeElement(ACEParams(adc=adc))
        for ob in (2, 4, 8):
            c = a.mvm_cost(64, 64, input_bits=8, output_bits=ob, value_bits=1)
            print(f"{name:<6}{ob:>9}{c.cycles:>9}{c.adc_cycles:>9}"
                  f"{100*c.adc_fraction:>7.0f}%{c.energy_pJ:>12.1f}")

    print("\nResNet-20 needs ~8 output bits, where SAR has the edge over ramp.")
