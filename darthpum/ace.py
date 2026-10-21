"""Analog Compute Element: ReRAM crossbar matrix-vector multiply.

How the analog multiply works
-----------------------------
Matrix values are programmed as device conductances. Driving wordline i with
voltage V_i makes each device pass I = V_i * G_ij (Ohm), and every device on
bitline j sums its current onto that wire (Kirchhoff). The current on bitline j is
therefore sum_i V_i * G_ij -- a dot product computed by physics, in one shot,
for all 64 bitlines at once. That is the entire appeal: no clocked multiply-
accumulate, just settling time.

The costs are at the edges. Inputs must be converted to voltages, outputs back to
digits, and analog devices hold only a few bits reliably. So:

  * **Matrix bit-slicing.** A W-bit matrix value is split across
    ceil(W / cell_bits) arrays, and their partial products recombined by shifting
    and adding ([DARTH] Figure 2). In SLC mode (1 bit/cell) an 8-bit weight needs
    8 arrays.
  * **Input bit-slicing.** Rather than build high-resolution DACs, an N-bit input
    is applied one bit per cycle over N cycles, each producing a partial product
    ([DARTH] Section 2.2.1). This also removes DAC non-linearity as an error source.

Where the ADC choice bites
--------------------------
Digitising the bitlines dominates, and the two converter types fail in opposite
directions:

  * **SAR** does a binary search: fast per conversion, but converts one bitline at
    a time, so a handful of ADCs must be multiplexed across all 64 bitlines.
    Cost scales with *bitline count*: ceil(64 / n_adc) cycles.
  * **Ramp** sweeps a reference voltage and compares, which is slow -- 256 cycles
    at full 8-bit resolution -- but it digitises **every bitline concurrently**,
    and it can stop early when only the low-order output bits are wanted. Cost
    scales with *required output precision*: 2^out_bits cycles.

So the better converter is a property of the workload, not of the chip: a
kernel that only needs a couple of output bits favours ramp's early
termination, while ResNet-20's ~8-bit outputs cost ramp the full 256 cycles
against SAR's 32 -- SAR wins by 8x, matching the paper's overall 1.5x
advantage for SAR once averaged across its workload mix ([DARTH] Section 7.3).

The paper picks one converter for the whole chip and accepts the loss on
workloads that would have preferred the other. Modelling both explicitly
here is what lets us measure that loss.
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
    adc_cycles: int
    """Cycles spent in conversion alone -- usually most of `cycles`."""

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

    # -- mapping --------------------------------------------------------

    def tiles_for_matrix(self, rows: int, cols: int, value_bits: int = 8) -> int:
        """Arrays needed to hold a rows x cols matrix.

        The standard ISAAC/PUMA tiling: chop the matrix into array-sized blocks,
        zero-padding the partial ones, then replicate across bit-slices.
        Row-tiles produce partial sums that must be accumulated digitally.
        """
        slices = math.ceil(value_bits / self.p.cell_bits)
        return (math.ceil(rows / self.p.array.rows)
                * math.ceil(cols / self.p.array.cols)
                * slices)

    def fits(self, rows: int, cols: int, value_bits: int = 8) -> bool:
        return self.tiles_for_matrix(rows, cols, value_bits) <= self.p.num_arrays

    # -- conversion cost ------------------------------------------------

    def adc_cycles_per_slice(self, output_bits: int | None = None) -> int:
        """Cycles to digitise one array's bitlines after a single input bit-slice.

        This is the method that encodes the whole SAR-vs-ramp trade-off, so it is
        also the hook our contribution modifies.
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

    # -- cost -----------------------------------------------------------

    def mvm_cost(
        self,
        rows: int,
        cols: int,
        input_bits: int = 8,
        output_bits: int | None = None,
        value_bits: int = 8,
    ) -> MVMCost:
        """Cost of multiplying a rows x cols matrix by one input vector.

        Row-tiles are summed digitally afterwards, by the DCE -- that cost belongs
        to the coordination layer (hct.py), not here.
        """
        output_bits = (output_bits if output_bits is not None
                       else self.p.adc.resolution_bits)

        tiles = self.tiles_for_matrix(rows, cols, value_bits)
        col_tiles = math.ceil(cols / self.p.array.cols)
        slices = math.ceil(value_bits / self.p.cell_bits)

        # Arrays operate in parallel, limited by how many the ACE has.
        waves = math.ceil(tiles / self.p.num_arrays)

        per_slice = 1 + self.adc_cycles_per_slice(output_bits)  # drive + convert
        adc_only = self.adc_cycles_per_slice(output_bits)

        cycles = waves * input_bits * per_slice
        adc_cycles = waves * input_bits * adc_only

        active = min(tiles, self.p.num_arrays)
        energy = self._energy_pJ(cycles, adc_cycles, active)

        return MVMCost(
            rows=rows, cols=cols,
            input_bits=input_bits, output_bits=output_bits,
            arrays_used=tiles, cycles=cycles,
            energy_pJ=energy, adc_cycles=adc_cycles,
        )

    def _energy_pJ(self, cycles: int, adc_cycles: int, active_arrays: int) -> float:
        """Table 3 power x active time.

        Row periphery and sample-and-hold burn for the whole operation; the ADCs
        only while converting.
        """
        ns = cycles / self.chip.clock_GHz
        adc_ns = adc_cycles / self.chip.clock_GHz

        array_mW = (POWER_MW["Row Periphery"] + POWER_MW["S&H (Analog)"]) * active_arrays
        adc_key = "SAR ADC" if self.p.adc.kind == "SAR" else "Ramp ADC"
        adc_mW = POWER_MW[adc_key] * self.p.adc.count

        return (array_mW * ns + adc_mW * adc_ns)  # mW*ns == pJ

    # -- functional -----------------------------------------------------

    def mvm(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
        output_bits: int | None = None,
        ideal: bool = True,
    ) -> np.ndarray:
        """Functionally evaluate matrix @ vector as the hardware would.

        With ideal=True this is exact, and must agree with numpy -- that identity
        is what the test suite checks, so an error in bit-slicing cannot hide.
        Device non-idealities are applied separately via noise.py (CrossSim), so
        that the timing model and the accuracy model stay independently testable.
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
        """Evaluate via explicit input bit-slicing, exactly as the hardware does.

        Applies one input bit per cycle and recombines the partial products by
        shifting. The result must equal the direct product -- this is the check
        that our understanding of the dataflow is right, not merely plausible.
        """
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
