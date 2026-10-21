"""Area and power model for a DARTH-PUM Hybrid Compute Tile (HCT).

This is the first thing to build and the first thing to validate, because it fixes
how many tiles fit on the chip -- and every throughput number in the paper is an
iso-area comparison, so the tile count multiplies straight into the headline results.

All component areas are [DARTH] Table 3. The model is validated by checking that it
reproduces the tile counts the paper states in Section 6.
"""

from dataclasses import dataclass

from .device import ACEParams, ADCParams, ChipParams, DCEParams


# --------------------------------------------------------------------------
# [DARTH] Table 3 -- area, um^2
# --------------------------------------------------------------------------

DCE_AREA_UM2 = {
    "ReRAM Array": 240.0,
    "Pipeline Control": 74_000.0,
    "IO Ctrl": 9_600.0,
    "Decode & Drive": 280.0,
    "Pipeline Select": 64.0,
}

ACE_AREA_UM2 = {
    "ReRAM Array": 240.0,
    "Input Buffers": 27_000.0,
    "Row Periphery": 13_000.0,
    "Sample & Hold": 62.0,
}

HCT_GLUE_AREA_UM2 = {
    "Shift Unit": 946.0,
    "Transpose Unit": 1_760.0,
    "A/D Arbiter": 0.6,
    "Instr. Injection Unit": 42.0,
}

ADC_AREA_UM2 = {"SAR": 600.0, "RAMP": 3_800.0}

FRONTEND_AREA_UM2 = 87_000.0
FRONTEND_POWER_MW = 63.0


# --------------------------------------------------------------------------
# [DARTH] Table 3 -- power, mW
# --------------------------------------------------------------------------

POWER_MW = {
    "Array (Bool Ops)": 8.0,
    "Row Periphery": 0.7,
    "Pipeline Ctrl": 1.6,
    "SAR ADC": 1.5,
    "S&H (Analog)": 2.1e-5,
    "Ramp ADC": 1.2,
}


# --------------------------------------------------------------------------

@dataclass
class AreaBreakdown:
    """Per-HCT area, decomposed so the report can show where it goes."""

    dce_um2: float
    ace_um2: float
    glue_um2: float
    adc_um2: float
    frontend_share_um2: float

    @property
    def total_um2(self) -> float:
        return (self.dce_um2 + self.ace_um2 + self.glue_um2
                + self.adc_um2 + self.frontend_share_um2)

    def as_dict(self) -> dict:
        return {
            "DCE": self.dce_um2,
            "ACE": self.ace_um2,
            "HCT glue": self.glue_um2,
            "ADC": self.adc_um2,
            "Front end (1/8)": self.frontend_share_um2,
            "TOTAL": self.total_um2,
        }


def hct_area(
    dce: DCEParams | None = None,
    ace: ACEParams | None = None,
    chip: ChipParams | None = None,
) -> AreaBreakdown:
    """Area of one Hybrid Compute Tile.

    The ReRAM array entries in Table 3 are per-element totals (240 um^2 covers all
    4096 arrays of a DCE), not per-array -- that is the only reading under which
    the table reconciles with the paper's stated tile count. See reconcile().
    """
    dce = dce or DCEParams()
    ace = ace or ACEParams()
    chip = chip or ChipParams()

    adc_area = ADC_AREA_UM2[ace.adc.kind] * ace.adc.count

    return AreaBreakdown(
        dce_um2=sum(DCE_AREA_UM2.values()),
        ace_um2=sum(ACE_AREA_UM2.values()),
        glue_um2=sum(HCT_GLUE_AREA_UM2.values()),
        adc_um2=adc_area,
        frontend_share_um2=FRONTEND_AREA_UM2 / chip.hcts_per_frontend,
    )


def hct_count(
    dce: DCEParams | None = None,
    ace: ACEParams | None = None,
    chip: ChipParams | None = None,
) -> int:
    """How many HCTs fit in the iso-area die budget."""
    chip = chip or ChipParams()
    return int(chip.die_area_um2 // hct_area(dce, ace, chip).total_um2)


def hct_capacity_bits(
    dce: DCEParams | None = None,
    ace: ACEParams | None = None,
) -> int:
    """Memory capacity of one HCT, in bits (SLC)."""
    dce = dce or DCEParams()
    ace = ace or ACEParams()
    return dce.arrays * dce.array.bits + ace.num_arrays * ace.array.bits


def chip_capacity_GB(
    dce: DCEParams | None = None,
    ace: ACEParams | None = None,
    chip: ChipParams | None = None,
) -> float:
    n = hct_count(dce, ace, chip)
    return n * hct_capacity_bits(dce, ace) / 8 / 2**30


# --------------------------------------------------------------------------
# Validation against the paper
# --------------------------------------------------------------------------

PAPER_CLAIMS = {
    # [DARTH] Section 6: "With SAR ADCs, an iso-area DARTH-PUM chip contains 1860
    # HCTs (total memory: 4.1 GB), while with ramp ADCs it contains 1660 HCTs (3.7 GB)."
    "SAR": {"hcts": 1860, "GB": 4.1},
    "RAMP": {"hcts": 1660, "GB": 3.7},
}


def reconcile() -> dict:
    """Compare our area model against the paper's stated tile counts.

    Result (see tests/test_area.py):

      SAR  -- we get 1845 vs the paper's 1860, a 0.8% error. The model reproduces
              the paper. Good.

      RAMP -- we get 1811 vs the paper's 1660, a 9.1% error, and in the wrong
              direction: our chip holds *more* tiles than theirs, meaning their
              ramp configuration costs more area than Table 2 + Table 3 imply.
              Solving for the missing area gives ~16.7k um^2 of ADC per HCT, i.e.
              about 4 ramp ADCs per ACE rather than the 1 printed in Table 2.

    We report the discrepancy rather than silently fitting to it. The SAR
    configuration is the paper's default for all of Section 7 except 7.3, so the
    headline results rest on the number we do reproduce.
    """
    out = {}
    for kind, adc in (("SAR", ADCParams.sar()), ("RAMP", ADCParams.ramp())):
        ace = ACEParams(adc=adc)
        n = hct_count(ace=ace)
        gb = chip_capacity_GB(ace=ace)
        claim = PAPER_CLAIMS[kind]
        out[kind] = {
            "hct_area_um2": hct_area(ace=ace).total_um2,
            "hcts_model": n,
            "hcts_paper": claim["hcts"],
            "hcts_error_pct": 100 * (n - claim["hcts"]) / claim["hcts"],
            "GB_model": gb,
            "GB_paper": claim["GB"],
            "GB_error_pct": 100 * (gb - claim["GB"]) / claim["GB"],
        }

    # What ADC count would be needed to land on the paper's ramp number?
    chip = ChipParams()
    target = chip.die_area_um2 / PAPER_CLAIMS["RAMP"]["hcts"]
    base = hct_area(ace=ACEParams(adc=ADCParams(kind="RAMP", count=0))).total_um2
    out["RAMP"]["implied_adc_count"] = (target - base) / ADC_AREA_UM2["RAMP"]

    return out


if __name__ == "__main__":
    import json
    print(json.dumps(reconcile(), indent=2))
