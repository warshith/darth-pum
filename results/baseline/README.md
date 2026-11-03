# Baseline Results

Reproduction of DARTH-PUM's published ResNet-20-on-CIFAR-10 numbers against
a CPU-plus-analog-PUM Baseline. See `paper/report.tex` Sections III-IV for
the full methodology.

| Metric | Paper | Ours | Agreement |
|---|---|---|---|
| Tile count (iso-area) | 1860 | 1844 | 0.86% |
| Throughput speedup | 14.8x | 14.72x | 0.5% |
| Latency speedup | 1.667x | 1.67x | <0.1% |
| Accuracy, fp32 | - | 91.69% | - |
| Accuracy, ideal analog (no noise) | - | 91.69% | - |
| Accuracy, with device noise (1 bit/cell) | 75.4% | 83.43% | see report |

## Figures

- `reproduction_summary.pdf` -- throughput/latency speedup, paper vs. ours
- `resnet20_accuracy.pdf` -- accuracy under increasing analog non-idealities
- `layer_footprint.pdf` -- physical array footprint per ResNet-20 layer
- `oscar_crosscheck.pdf` -- digital cost model validated against RACER/OSCAR

Regenerate with `python -m experiments.make_figures`.
