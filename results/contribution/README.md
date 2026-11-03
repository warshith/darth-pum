# Contribution Results

Two small architectural extensions, each measured end to end rather than at
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

Regenerate with `python -m experiments.make_figures`.
