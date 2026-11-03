"""Chip-level model: lots of hybrid tiles, scheduling between them, throughput.

Everything in hct.py gives you *latency* for one tile doing one thing. The
numbers the paper actually reports are chip-level throughput, so this module
is what turns "how long does one HCT take" into something comparable.

Roughly: an iso-area DARTH-PUM chip holds ~1844 tiles (area.py). A workload
instance (say, one ResNet-20 inference) occupies some tiles for some cycles.
Run a bunch of instances side by side and throughput falls out:

    instances in parallel = tiles available / tiles per instance
    throughput            = instances in parallel / latency per instance

Footprint matters as much as raw speed because of this. ResNet-20's biggest
layer only needs 9 of an ACE's 64 arrays, so the whole network fits in a
couple of tiles and the chip runs ~900 inferences at once. A workload needing
500 tiles per instance would only get 3 running in parallel, however fast each
one was -- which is more or less [DARTH] Section 7.1's argument against
AppAccel: it spends area on special-function units, so it runs fewer
instances in parallel, and DARTH-PUM makes that gap back on density alone.

Area isn't the only budget though. ReRAM pulls real current when it switches
and a chip can't just fire every array at once. MASTODON's shipped configs
carry watt_thermal_limit=80, and Section 6 of the paper notes the DigitalPUM
comparison point only runs "two pipelines active per cluster to stay within
thermal limits" -- a 32x derate on a 64-pipeline element. Skipping this would
make every digital configuration look better than it should, so the cap is
modelled explicitly here instead of assumed away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .area import chip_capacity_GB, hct_count
from .device import ACEParams, ChipParams, DCEParams
from .hct import HybridComputeTile

DEFAULT_THERMAL_LIMIT_W = 80.0  # MASTODON's shipped configs: watt_thermal_limit=80


@dataclass
class Workload:
    """One instance of an application, costed on a single tile.

    work_units exists so throughput can be reported in whatever unit the
    paper uses for that workload -- inferences for ResNet-20, etc.
    """

    name: str
    tiles_per_instance: int
    cycles_per_instance: int
    energy_pJ_per_instance: float
    work_units: float = 1.0
    unit: str = "instances"
    breakdown: dict[str, int] = field(default_factory=dict)  # per-kernel cycles, for Fig-14-style charts

    def __post_init__(self):
        if self.tiles_per_instance < 1:
            raise ValueError(f"{self.name}: needs at least one tile")
        if self.cycles_per_instance < 1:
            raise ValueError(f"{self.name}: needs at least one cycle")


@dataclass(frozen=True)
class ChipResult:
    """What the chip achieves on a workload."""

    workload: str
    tiles_total: int
    tiles_per_instance: int
    instances_parallel: int
    latency_ns: float
    throughput_per_s: float
    energy_pJ_per_unit: float
    power_W: float
    thermally_limited: bool
    unit: str

    def as_dict(self) -> dict:
        return {
            "workload": self.workload,
            "tiles_total": self.tiles_total,
            "tiles_per_instance": self.tiles_per_instance,
            "instances_parallel": self.instances_parallel,
            "latency_ns": self.latency_ns,
            f"throughput_{self.unit}_per_s": self.throughput_per_s,
            "energy_pJ_per_unit": self.energy_pJ_per_unit,
            "power_W": self.power_W,
            "thermally_limited": self.thermally_limited,
        }


class Chip:
    """A full DARTH-PUM chip: N hybrid tiles under an area and a power budget."""

    def __init__(
        self,
        ace_params: ACEParams | None = None,
        dce_params: DCEParams | None = None,
        chip_params: ChipParams | None = None,
        optimized: bool = True,
        thermal_limit_W: float = DEFAULT_THERMAL_LIMIT_W,
        name: str = "DARTH-PUM",
    ):
        self.name = name
        self.ace_params = ace_params or ACEParams()
        self.dce_params = dce_params or DCEParams()
        self.p = chip_params or ChipParams()
        self.thermal_limit_W = thermal_limit_W
        self.tile = HybridComputeTile(
            self.ace_params, self.dce_params, self.p, optimized=optimized
        )

    @property
    def tiles(self) -> int:
        """Tiles fitting the iso-area die budget."""
        return hct_count(self.dce_params, self.ace_params, self.p)

    @property
    def capacity_GB(self) -> float:
        return chip_capacity_GB(self.dce_params, self.ace_params, self.p)

    def run(self, w: Workload) -> ChipResult:
        """Throughput, energy and power for a workload on this chip."""
        if w.tiles_per_instance > self.tiles:
            raise ValueError(
                f"{w.name} needs {w.tiles_per_instance} tiles but the chip has "
                f"{self.tiles}. It would have to be executed in passes, which "
                "this model does not yet handle."
            )

        wanted = self.tiles // w.tiles_per_instance
        seconds = w.cycles_per_instance / (self.p.clock_GHz * 1e9)

        # power if every copy ran at the same time
        watts_each = (w.energy_pJ_per_instance * 1e-12) / seconds
        full_power = wanted * watts_each

        throttled = full_power > self.thermal_limit_W
        if throttled:
            parallel = max(1, int(self.thermal_limit_W // watts_each))
        else:
            parallel = wanted

        return ChipResult(
            workload=w.name,
            tiles_total=self.tiles,
            tiles_per_instance=w.tiles_per_instance,
            instances_parallel=parallel,
            latency_ns=seconds * 1e9,
            throughput_per_s=parallel * w.work_units / seconds,
            energy_pJ_per_unit=w.energy_pJ_per_instance / w.work_units,
            power_W=parallel * watts_each,
            thermally_limited=throttled,
            unit=w.unit,
        )

    @staticmethod
    def speedup(result: ChipResult, baseline: ChipResult) -> float:
        """Throughput of `result` relative to `baseline`, the paper's metric."""
        return result.throughput_per_s / baseline.throughput_per_s

    @staticmethod
    def energy_saving(result: ChipResult, baseline: ChipResult) -> float:
        """Energy reduction factor, reported the paper's way (bigger = better)."""
        return baseline.energy_pJ_per_unit / result.energy_pJ_per_unit

    @staticmethod
    def geomean(values: list[float]) -> float:
        """[DARTH] reports geometric means across its three workloads."""
        if not values:
            raise ValueError("geomean of nothing")
        if any(v <= 0 for v in values):
            raise ValueError(f"geomean needs positive values, got {values}")
        return math.exp(sum(math.log(v) for v in values) / len(values))


if __name__ == "__main__":
    from .device import ADCParams

    for adc in (ADCParams.sar(), ADCParams.ramp()):
        c = Chip(ace_params=ACEParams(adc=adc))
        print(f"{adc.kind:<5} {c.tiles:>5} tiles   {c.capacity_GB:.2f} GB")

    print()
    chip = Chip()
    # placeholder workload until the real ones are wired in -- one 64x64 MVM/tile
    cost = chip.tile.mvm(64, 64, input_bits=8, value_bits=1)
    w = Workload(
        name="single-MVM",
        tiles_per_instance=1,
        cycles_per_instance=cost.cycles,
        energy_pJ_per_instance=cost.energy_pJ,
        unit="MVMs",
    )
    r = chip.run(w)
    print(f"{r.workload}: {r.instances_parallel} parallel, "
          f"{r.latency_ns:.0f} ns each")
    print(f"  throughput {r.throughput_per_s/1e9:.2f} G{r.unit}/s, "
          f"power {r.power_W:.1f} W, thermally limited: {r.thermally_limited}")
