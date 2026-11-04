"""Does the DCE actually have to be built on ReRAM?

Everything the HCT's coordination layer does -- the shift unit, the transpose
unit, the arbiter -- is bookkeeping: cycle counts, bit positions, which op is
waiting on which. None of that logic ever looks at what the cell underneath
is physically made of, only at its access latency and how many bits it holds.
That's a narrow interface, which means the two compute elements don't have to
share one device technology just because the paper happens to build them that
way.

The ACE can't really move off ReRAM: analog MVM needs a cell that holds many
distinguishable conductance levels, and that's a ReRAM strength, not an
STT-MRAM one (TMR ratio is only ~2-3x, nowhere near ReRAM's ~500x). But the
DCE only ever reads two states and gets rewritten on almost every cycle it's
active -- and endurance, not dynamic range, is what a component used like
that actually needs. STT-MRAM's endurance is routinely cited several orders
of magnitude above ReRAM's, which is exactly the property a bit-pipelined
Boolean element would want.

One honest caveat this swap rests on: RACER's digital PUM doesn't just store
and read bits in the DCE, it computes NOR gates *inside* the array via the
device's own switching threshold ("stateful logic", MAGIC/IMPLY-style). That
has been demonstrated on STT-MTJ devices in simulation, but the same small
resistance ratio that's fine for plain storage is a real, still-open
reliability problem for doing logic this way -- published work reports
HRS/LRS overlap and read-disturb error rates that need dedicated circuit
work to bring down. So treat this as a proposed direction grounded in real
published attempts, not a settled fact: we measure the tradeoff assuming an
equivalent logic scheme is realized on MRAM, not claiming it already is.

So: keep the ACE on ReRAM, swap just the DCE's device model to
DeviceParams.stt_mram() (darthpum/device.py), and measure what that swap
actually buys -- a heterogeneous-memory HCT, one device per compute element
instead of one for the whole tile. Two numbers come out of this file:

  1. switching energy per op, which DigitalComputeElement.switching_energy_pJ
     already reports per-device once DCEParams carries a device (see dce.py)
     -- no new code needed there, just a different device plugged in.

  2. how long one HCT's DCE lasts running ResNet-20 back to back, under each
     device's cited endurance. This isn't a full wear-leveling simulation --
     we don't model how RACER's scheduler actually distributes writes across
     a DCE's 16.7M cells -- so we assume even wear across all of them, which
     is the standard first-order way this kind of lifetime number gets
     estimated, and it's the right comparison for isolating what the device
     swap alone changes rather than how well-scheduled the chip is.

Numbers, ResNet-20, one HCT's DCE, non-MVM ops only (BN fold + ReLU + residual
add -- the MVM itself lives on the ACE and isn't part of this swap):

    device      switch energy   endurance   avg writes/cell/inference   lifetime
    ReRAM       0.0128 pJ       1e6 writes  1.19e-2                     ~0.04 yr  (~15 days)
    STT-MRAM    1.05 pJ         1e12 writes 1.19e-2                     ~36,200 yr

Same op mix hits every cell either way, so the lifetime gap is just the
endurance ratio we put in -- the swap buys that almost for free. The energy
side is a real, not-small cost: 1.05 pJ comes from an actual NVSim run on its
own shipped STT-RAM sample (see DeviceParams.stt_mram()'s docstring), and
it's ~82x ReRAM's per-switching-event energy, not the mild "couple-x" cost a
first guess might suggest. That 82x stays contained, though: it only shows up
in switching_energy_pJ, the bottom-up cross-check. The chip's real,
everywhere-else-reported DCE power comes from Table 3's whole-element active
power (dce.py's _energy_pJ), which is dominated by pipeline control logic,
not per-switch energy -- so this device swap's real cost is honestly large
at the device level and still doesn't move the headline throughput/energy
numbers, which is itself worth stating plainly rather than picking whichever
framing sounds better. Run this file directly to regenerate both numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from darthpum.dce import DigitalComputeElement
from darthpum.device import DCEParams, DeviceParams

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def dce_switching_events_per_inference(dce: DigitalComputeElement) -> int:
    """Every BN fold, ReLU and residual add ResNet-20 needs, summed layer by
    layer, counted as raw cell-switching events rather than energy -- that's
    what wears a cell out, not how many pJ it cost. Same op pattern as
    resnet_pum.nonmvm_cost; the MVM itself doesn't touch the DCE at all, so
    it's left out here on purpose."""
    from workloads.resnet20.mapping import map_resnet20
    from workloads.resnet20.resnet_pum import OUTPUT_BITS, VALUE_BITS

    mappings = map_resnet20(value_bits=VALUE_BITS)
    total = 0
    for m in mappings:
        if m.kind == "fc":
            continue  # no BN/ReLU after the final classifier
        elements = m.out_shape[0]
        total += dce.switching_events("ADD", OUTPUT_BITS, elements)       # BN fold
        total += dce.switching_events("RELUv1", OUTPUT_BITS, elements)    # ReLU
        if m.name.endswith(".conv2"):
            total += dce.switching_events("ADD", OUTPUT_BITS, elements)   # residual
    return total


def cycles_per_inference() -> int:
    """Total HCT cycles (ACE + DCE) for one ResNet-20 inference, default
    config. Used only to turn "inferences to wearout" into wall-clock time."""
    from workloads.resnet20.resnet_pum import inference_cost

    mvm_costs, nonmvm_costs = inference_cost()
    return sum(c.cycles for c in mvm_costs + nonmvm_costs)


@dataclass(frozen=True)
class EnduranceResult:
    device_name: str
    switch_energy_pJ: float
    endurance_writes: float
    cells_per_dce: int
    switching_events_per_inference: int
    inferences_to_wearout: float
    seconds_to_wearout: float

    @property
    def avg_writes_per_cell_per_inference(self) -> float:
        return self.switching_events_per_inference / self.cells_per_dce

    @property
    def years_to_wearout(self) -> float:
        return self.seconds_to_wearout / SECONDS_PER_YEAR


def endurance_margin(device: DeviceParams, device_name: str) -> EnduranceResult:
    """One HCT's DCE, running ResNet-20 back to back forever, assuming perfect
    even wear across every cell it owns -- how long before the busiest cell
    would, on average, have worn out."""
    dce = DigitalComputeElement(DCEParams(device=device))
    events = dce_switching_events_per_inference(dce)
    cells = (dce.p.num_pipelines * dce.p.pipeline_depth
             * dce.p.array.rows * dce.p.array.cols)
    avg_writes_per_cell = events / cells
    inferences = device.endurance_writes / avg_writes_per_cell

    cycles = cycles_per_inference()
    seconds_per_inference = cycles / (dce.chip.clock_GHz * 1e9)
    seconds = inferences * seconds_per_inference

    return EnduranceResult(
        device_name=device_name,
        switch_energy_pJ=device.switch_energy_pJ,
        endurance_writes=device.endurance_writes,
        cells_per_dce=cells,
        switching_events_per_inference=events,
        inferences_to_wearout=inferences,
        seconds_to_wearout=seconds,
    )


def compare_devices() -> dict[str, EnduranceResult]:
    return {
        "ReRAM": endurance_margin(DeviceParams(), "ReRAM (default)"),
        "STT-MRAM": endurance_margin(DeviceParams.stt_mram(), "STT-MRAM"),
    }


if __name__ == "__main__":
    results = compare_devices()
    print(f"{'device':<12}{'switch pJ':>10}{'endurance':>12}{'avg wr/cell/inf':>17}"
          f"{'inf to wearout':>16}{'lifetime (yr)':>15}")
    for r in results.values():
        print(f"{r.device_name:<12}{r.switch_energy_pJ:>10.4f}{r.endurance_writes:>12.0e}"
              f"{r.avg_writes_per_cell_per_inference:>17.3e}"
              f"{r.inferences_to_wearout:>16.3e}{r.years_to_wearout:>15.2f}")

    rr, mr = results["ReRAM"], results["STT-MRAM"]
    print(f"\nSTT-MRAM costs {mr.switch_energy_pJ / rr.switch_energy_pJ:.2f}x more pJ "
          f"per switching event than ReRAM, but lasts "
          f"{mr.years_to_wearout / rr.years_to_wearout:.0f}x longer under the same "
          "sustained non-MVM workload.")
