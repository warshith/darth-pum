# Contribution Results

Three small architectural extensions, each measured end to end rather than at
a more flattering isolated-kernel or naive-estimate figure. See
`paper/report.tex` Section V for the full discussion.

## Row-level zero-skipping

| | |
|---|---|
| Measured activation sparsity | 51.1% mean, across 19 ReLU sites |
| Naive estimate (isolated kernel) | ~48% analog energy reduction |
| Measured (end to end) | 3.07% analog energy reduction |

Figure: `zero_skip_dilution.pdf`

## Precision / throughput tradeoff

| cell_bits | Tiles/instance | Throughput | Speedup | Accuracy |
|---|---|---|---|---|
| 1 (SLC) | 13 | 10,364.7 inf/s | 1.00x | 83.43% |
| 2 | 7 | 19,453.5 inf/s | 1.88x | 70.68% |
| 4 | 4 | 34,099.1 inf/s | 3.29x | 53.84% |

Figure: `precision_tradeoff.pdf`

## Heterogeneous-memory HCT: ReRAM vs. STT-MRAM DCE

Proposed direction, not a settled claim: assumes an equivalent in-memory
logic scheme can be realized on STT-MRAM, which published work has
demonstrated in simulation but not fully solved for reliability. Device
numbers (switching energy, latency) are from a real NVSim run on its own
shipped STT-RAM sample, not a hand-picked figure.

| | ReRAM | STT-MRAM |
|---|---|---|
| Switching energy | 0.0128 pJ | 1.05 pJ |
| Endurance (cited) | 1e6 writes | 1e12 writes |
| DCE lifetime, ResNet-20 back to back, even wear | ~0.04 yr (~2 weeks) | ~36,200 yr |

Figure: `heterogeneous_memory.pdf`

Regenerate with `python -m experiments.make_figures`.
