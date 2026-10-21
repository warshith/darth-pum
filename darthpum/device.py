"""Technology and hardware constants for the DARTH-PUM model.

Every number here is traceable to a published source. The `src` field on each
constant names that source so the report can cite it and so a reviewer can check
we did not invent anything.

Sources
-------
[DARTH]  Wong, Feinberg & Ghose. "DARTH-PUM: A Hybrid Processing-Using-Memory
         Architecture." ASPLOS '26, pp. 563-582.  Tables 2 and 3, Section 6.
[RACER]  Truong et al. "RACER: Bit-Pipelined Processing Using Resistive Memory."
         MICRO '21.  Table 1 (device), Table 4 (per-cluster area/latency/power).
[OSCAR]  Truong et al. "Adapting the RACER Architecture to Integrate Improved
         In-ReRAM Logic Primitives." JETCAS '22.
"""

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# ReRAM device
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceParams:
    """Resistive device characteristics."""

    switch_energy_pJ: float = 0.0128
    """Energy per switching transition. [RACER] Table 1, 'conservative' column."""

    switch_latency_ns: float = 1.0
    """Switching latency of the resistive element. [RACER] Section 3."""

    on_off_ratio: float = 500.0
    """R_off / R_on. [RACER] Table 1."""

    tech_node_nm: int = 15
    """FreePDK 15 nm. [RACER] Section 7; [DARTH] Section 6 scales all area to 15 nm."""


# --------------------------------------------------------------------------
# Array geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ArrayParams:
    """A single ReRAM crossbar array.

    64x64 is a deliberate choice, not a tuning knob: [RACER] Section 3 caps array
    dimension at n <= 1024 because of wire resistance, sneak-path leakage and wire
    current limits. [DARTH] Table 2 uses 64x64 for both the digital and analog sides.
    """

    rows: int = 64
    cols: int = 64

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    @property
    def bits(self) -> int:
        """Capacity in bits when used as single-level (1 bit/cell) storage."""
        return self.cells


# --------------------------------------------------------------------------
# Digital Compute Element (DCE) -- RACER bit-pipelined digital PUM
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DCEParams:
    """[DARTH] Table 2, '1 Digital Compute Element'."""

    num_pipelines: int = 64
    """Independent bit-pipelines per DCE."""

    pipeline_depth: int = 64
    """Arrays per pipeline. Also the maximum operand bit-width, since an N-bit
    value is bit-striped one bit position per array. [RACER] Section 4."""

    array: ArrayParams = field(default_factory=ArrayParams)

    scan_chain_cycles: int = 8
    """Cycles for a micro-op to propagate tile-to-tile along the scan chain.
    The control unit therefore dispatches a new micro-op set every 8 cycles.
    [RACER] Section 5."""

    buffer_write_cycles: int = 2
    """Writing a column into a buffer: cycle 1 presets to all-zero, cycle 2
    writes the 1s. [RACER] Section 4."""

    @property
    def arrays(self) -> int:
        return self.num_pipelines * self.pipeline_depth


# --------------------------------------------------------------------------
# Analog Compute Element (ACE) -- crossbar MVM
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ADCParams:
    """Analog-to-digital converter.

    The two types trade off very differently, and [DARTH] Section 7.3 shows the
    better choice is workload-dependent, so both are modelled explicitly here
    rather than averaged into one number.

    SAR  : binary search, log(N) comparisons, fast, but a single conversion at a
           time -- so the few ADCs must be multiplexed across the 64 bitlines.
    Ramp : linearly sweeps a reference over N increments. Slow in the worst case,
           but converts all bitlines *concurrently*, and can terminate early when
           only the low-order output bits are needed.
    """

    kind: str = "SAR"
    count: int = 2
    latency_cycles: int = 1
    resolution_bits: int = 8

    @staticmethod
    def sar() -> "ADCParams":
        """[DARTH] Table 2: SAR, 2 ADCs, 1 cycle latency."""
        return ADCParams(kind="SAR", count=2, latency_cycles=1)

    @staticmethod
    def ramp() -> "ADCParams":
        """[DARTH] Table 2: Ramp, 1 ADC, 256 cycle latency.

        NOTE: Table 2 says 1 ramp ADC, but that does not reconcile with the 1660
        HCT count in Section 6 -- see area.py:reconcile(). We expose the count so
        the discrepancy can be measured rather than papered over.
        """
        return ADCParams(kind="RAMP", count=1, latency_cycles=256)


@dataclass(frozen=True)
class ACEParams:
    """[DARTH] Table 2, '1 Analog Compute Element'."""

    num_arrays: int = 64
    array: ArrayParams = field(default_factory=ArrayParams)
    adc: ADCParams = field(default_factory=ADCParams.sar)

    cell_bits: int = 1
    """Bits stored per analog device. [DARTH] Section 7.5 reports the ResNet-20
    accuracy result in SLC (1 bit/cell) mode."""


# --------------------------------------------------------------------------
# Chip
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ChipParams:
    """[DARTH] Section 6."""

    clock_GHz: float = 1.0
    """DARTH-PUM runs at 1 GHz, raised from RACER's 333 MHz by using OSCAR."""

    die_area_um2: float = 2.57e8
    """2.57 cm^2 -- the die area of the Intel i7-13700 baseline CPU. All
    comparisons in the paper are iso-area to this."""

    hcts_per_frontend: int = 8
    """One fetch/decode/issue front end is shared by 8 HCTs. [DARTH] Table 3."""

    io_bytes_per_cycle: int = 8
    """ACE<->DCE data transfer network width, chosen to rate-match ADC throughput
    against DCE write bandwidth. [DARTH] Section 4."""


DEVICE = DeviceParams()
ARRAY = ArrayParams()
CHIP = ChipParams()
