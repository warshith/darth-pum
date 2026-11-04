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

    endurance_writes: float = 1.0e6
    """Write cycles before wearout. Commonly cited ReRAM range is ~1e6-1e9
    depending on the cell; we use the conservative end. Only used by the
    heterogeneous-memory extension (see contribution/heterogeneous_memory.py)
    -- nothing in the main timing/energy model reads this."""

    @staticmethod
    def stt_mram() -> "DeviceParams":
        """A second device option for the DCE, not the ACE: STT-MRAM only
        has two resistance states (on:off ratio ~2x, nowhere near ReRAM's
        500x), so it can't hold the multiple levels the ACE's multi-bit
        analog storage needs. But digital PUM already only ever reads two
        states, and endurance is the thing that actually matters for a
        component this heavily switched -- STT-MRAM's is routinely cited
        several orders of magnitude above ReRAM's.

        Important caveat, not just a formality: RACER's digital PUM doesn't
        just store and read bits, it computes NOR gates *inside* the array
        via the device's own switching threshold ("stateful logic" --
        MAGIC/IMPLY-style). That mechanism has been demonstrated on STT-MTJ
        devices in simulation, but the same small resistance ratio that's
        fine for plain storage is an open reliability problem for doing
        logic this way: published work reports real HRS/LRS overlap and
        read-disturb error rates that still need dedicated circuit work to
        bring down. So this device swap is a proposed direction grounded in
        real published attempts, not a settled, drop-in-safe claim -- we
        measure what the tradeoff would look like assuming an equivalent
        logic scheme is realized on MRAM, and say so plainly rather than
        asserting it as fact. This is the device knob the heterogeneous-
        memory extension turns -- the ACE keeps the chip's default ReRAM.

        switch_energy_pJ / switch_latency_ns / on_off_ratio below are from a
        real NVSim run (SEAL-UCSB/NVSim), not a hand-picked literature
        guess: NVSim's own shipped sample_STTRAM.cell gives
        ResistanceOn=3000ohm, ResistanceOff=6000ohm (on:off = 2.0x, even
        tighter than we'd first assumed, which only reinforces the
        reliability caveat above), SetEnergy=1pJ, SetPulse=10ns at the cell
        level. Running that cell in NVSim's sample_STTRAM_cache.cfg (22nm,
        8MB, 512-bit word) gives array-level write numbers -- 538.3 pJ and
        10.252 ns per 512-bit word, i.e. ~1.05 pJ/bit once real array
        overhead (decoders, mux, bitlines) is included, noticeably higher
        than the bare per-cell SetEnergy, which is the honest array-level
        number rather than the optimistic cell-level one. One more honest
        gap: NVSim's shipped STT-RAM sample is a 22nm config, not the
        chip's 15nm, since that's the sample NVSim ships -- tech_node_nm
        below reflects what was actually simulated, not what we'd need for
        an apples-to-apples node match.

        endurance_writes has no NVSim equivalent -- NVSim models
        timing/energy/area, not wearout, and grepping its source for
        endurance/retention/wearout turns up nothing -- so that figure stays
        a literature citation regardless: STT-MRAM endurance is commonly
        reported several orders of magnitude above ReRAM's, we use a
        representative value from that range."""
        return DeviceParams(
            switch_energy_pJ=1.05,
            switch_latency_ns=10.25,
            on_off_ratio=2.0,
            tech_node_nm=22,
            endurance_writes=1.0e12,
        )


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

    device: DeviceParams = field(default_factory=DeviceParams)
    """Which device backs this DCE's cells. Defaults to the chip's ReRAM.
    The heterogeneous-memory extension swaps this for DeviceParams.stt_mram()
    to measure what happens if the DCE (only the DCE -- the ACE needs
    ReRAM's multi-level storage and stays put) were built on STT-MRAM
    instead. Nothing else in this file changes when you do that; only the
    switching-energy and endurance numbers that actually depend on the
    underlying device respond."""

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
