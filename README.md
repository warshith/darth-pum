# DARTH-PUM: ResNet-20 on a Hybrid Processing-Using-Memory Architecture

A reproduction of parts of **"DARTH-PUM: A Hybrid Processing-Using-Memory
Architecture"** (Wong, Feinberg & Ghose — ASPLOS '26), plus two small
extensions.

## Background

Processing-using-memory (PUM) computes directly inside a ReRAM array instead
of moving data to a separate processor. There are two ways to do it:

- **Analog PUM** drives the array with voltages and reads the result off as
  current — Ohm's law and Kirchhoff's current law compute a matrix-vector
  multiply as physics, in one shot. Fast and efficient, but approximate, and
  it can only do multiply-accumulate.
- **Digital PUM** uses the same cells as one-bit-at-a-time Boolean logic
  gates. Exact and general, but much slower.

DARTH-PUM puts both on one chip and splits a workload between them:
multiplications go to the analog arrays, everything else goes to the digital
side.

## Scope

This project reproduces the ResNet-20 / CIFAR-10 part of the paper:

- An analog matrix-vector multiply, built on
  [CrossSim](https://github.com/sandialabs/cross-sim) (Sandia's open-source
  ReRAM crossbar simulator), including device programming noise.
- An operation-counting cost model for the digital, bit-serial in-memory
  logic (`darthpum/dce.py`), using real per-operation cycle counts from
  [MASTODON](https://github.com/ARCANA-Research/MASTODON)'s logic-family
  recipe book.
- ResNet-20 mapped so that convolutions run on the analog arrays and
  everything else (batch norm, ReLU, residual adds) runs on the digital side.
- A CPU-plus-analog-accelerator baseline, for comparison against the hybrid
  design.
- Accuracy under realistic device noise, measured end-to-end on the CIFAR-10
  test set.

## Results

| | |
|---|---|
| Tile count, iso-area chip | 1844 (paper: 1860, 0.9% off) |
| Throughput, hybrid vs. baseline | 14.72x (paper: 14.8x) |
| Latency, hybrid vs. baseline | 1.67x (paper: 1.667x) |
| Accuracy, fp32 | 91.69% |
| Accuracy, ideal analog (no device noise) | 91.69% |
| Accuracy, with device noise (1 bit/cell) | 83.43% |

## Two extensions

- **Zero-skipping.** Post-ReLU activations are ~51% zero on average
  (measured on real CIFAR-10 images). Disabling the analog row drivers for
  those zero inputs saves energy — reported at its real, measured,
  end-to-end figure.
- **Precision vs. throughput.** How many bits each analog cell stores is a
  dial, not a fixed choice. Sweeping it from 1 to 4 bits/cell gives up to
  3.29x more throughput, at a real and sizeable accuracy cost.

## Layout

```
darthpum/       device/area models, analog + digital compute elements,
                the hybrid coordination layer, chip-level scheduling
workloads/      ResNet-20 mapped onto the architecture
baselines/      the CPU + analog-accelerator comparison point
contribution/   the two extensions above
experiments/    figure-generation scripts
results/        baseline and contribution graphs/tables, standalone
tests/          correctness and reproduction checks
paper/          IEEE-format report
```

## Setup

```bash
./setup.sh      # clone CrossSim + MASTODON, create a venv, install deps
pytest tests/
```

## Team

| Name | Roll number |
|---|---|
| Grandhi Varshith | 230101041 |
| Gugulothu Maniprakash | 230101043 |
| Seela Mythresh Babu | 230101092 |
| Baswa Dhanush Sai | 230101024 |
