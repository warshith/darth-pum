# DARTH-PUM — Reproduction and Extension

A reproduction of **"DARTH-PUM: A Hybrid Processing-Using-Memory Architecture"**
(Ryan Wong, Ben Feinberg, Saugata Ghose — ASPLOS '26, pp. 563–582), plus a
proposed improvement.

## What the paper does

Computers spend most of their time and energy *moving* data between memory and the
processor rather than computing with it. Processing-using-memory (PUM) attacks this
by computing inside the memory array itself. ReRAM supports two very different ways
of doing that:

- **Analog PUM** exploits device physics directly. Drive the wordlines with
  voltages, and Ohm's law plus Kirchhoff's current law make the current on each
  bitline a dot product. Extremely fast and efficient — but approximate, and it can
  *only* do matrix–vector multiply.
- **Digital PUM** uses the same cells as Boolean logic gates. Exact and fully
  general, but slow: every operation is built up one bit at a time.

DARTH-PUM puts both on one die and makes them cooperate, so a single chip runs AES
encryption, a CNN, and a transformer encoder without the application-specific
fixed-function circuits that previous analog accelerators required. The hard part —
and the paper's real contribution — is the coordination between the two domains:
they compute on perpendicular axes, differ by orders of magnitude in latency, and
produce partial results that must be shifted and recombined.

## Reproduction strategy

The authors' simulator was **not released** (no artifact statement, no repository).
Both of its upstream components are public, however:

| Component | Source | License | Role here |
|---|---|---|---|
| **MASTODON** | [ARCANA-Research/MASTODON](https://github.com/ARCANA-Research/MASTODON) | MIT | Digital PUM. From the authors' own lab. We parse `src/bottom/ReRAM_oscar.h` for the OSCAR logic-family cost table. |
| **CrossSim** | [sandialabs/cross-sim](https://github.com/sandialabs/cross-sim) | BSD-3 | Analog crossbar accuracy: programming noise, read noise, drift, ADC/DAC quantization, parasitics. |
| **MILO** | — | — | ReRAM programming-noise model; already implemented inside CrossSim as `RRAMMilo`. |
| **RACER-Sim** | [ARCANA-Research/RACER-Artifacts](https://github.com/ARCANA-Research/RACER-Artifacts) | — | Earlier simulator; its cost tables cross-check ours. |

What is missing from all of these is exactly the hybrid coordination layer the paper
is about. **That is what this repository implements.** Upstream simulators are
cloned, not vendored — see `setup.sh`.

## Status

| Component | State |
|---|---|
| Area / power model (`darthpum/area.py`) | ✅ reproduces the paper's tile count to 0.9% |
| OSCAR cost table (`darthpum/recipes.py`) | ✅ 33 recipes parsed; cross-checks against published RACER figures |
| Digital compute element | ⬜ |
| Analog compute element | ⬜ |
| Hybrid coordination layer | ⬜ |
| AES-128 / ResNet-20 / LLM encoder | ⬜ |
| Baselines, contribution, report | ⬜ |

### Findings so far

**The area model reproduces, and exposes one inconsistency.** Summing [DARTH]
Table 3 and adding ⅛ of the shared front end gives 139 310 µm² per tile, so
**1844 tiles** fit the 2.57 cm² iso-area budget — against the paper's stated 1860,
a **0.9% error**. The ramp-ADC configuration does *not* reconcile: we get 1811
against their 1660. Solving for the shortfall implies **~4 ramp ADCs per analog
element**, not the 1 listed in Table 2. The SAR configuration is the paper's
default everywhere except Section 7.3, so the headline results rest on the number
that does reproduce.

**Our parsed OSCAR cost table cross-checks against the published figures.** An ADD
costs 21 cycles per bit-slice, of which 16 are NOR — fewer than the 22 NOR/bit that
RACER reports for the MAGIC family, with the saving coming from OSCAR's native
single-cycle OR. That is the expected shape, and it confirms we are reading the
recipe book correctly.

## Setup

```bash
./setup.sh          # clone upstream simulators, create venv, install deps
```

## Layout

```
darthpum/       simulator: device/area models, cost tables, compute elements,
                the hybrid coordination layer, chip-level scheduling
workloads/      AES-128, ResNet-20, LLM encoder — mapped onto the architecture
baselines/      CPU measurement harness + digital-only / analog-only chip models
experiments/    one script per paper figure; run_all.py regenerates everything
tests/          area model, AES vs NIST vectors, the Figure 9 MVM walkthrough
paper/          IEEE-format report (LaTeX + TikZ)
slides/         presentation
```
