"""Generate every result figure for the report, from live results.

Every number plotted here is either computed fresh from the simulator
modules, or pulled from a value already measured and pinned by the test
suite (CrossSim accuracy runs take minutes, so those three numbers are the
one exception -- see workloads/resnet20/accuracy.py and
contribution/precision_tradeoff.py for the runs that produced them).

The HCT architecture diagram is drawn directly in the report as a TikZ
figure, not generated here -- a block diagram has no data to compute.

Run:  python -m experiments.make_figures
"""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from experiments.plotstyle import (
    BLUE, BLUE_RAMP, FIGSIZE_SINGLE, FIGSIZE_WIDE, GREEN, NEUTRAL, ORANGE,
    bar_value_labels, save, setup,
)

os.makedirs("paper/figs", exist_ok=True)
setup()


# --------------------------------------------------------------------------
# Figure 1: reproduction summary -- throughput and latency, paper vs. ours.
# --------------------------------------------------------------------------

def fig_reproduction_summary():
    from darthpum.chip import Chip, Workload
    from darthpum.hct import HybridComputeTile
    from workloads.resnet20.resnet_pum import inference_cost, hcts_per_instance
    from baselines.resnet20_baseline import (
        ResNet20Baseline, CALIBRATED_LINK_GB_PER_S, CALIBRATED_ROUND_TRIP_NS,
    )

    tile = HybridComputeTile()
    mvm_costs, nonmvm_costs = inference_cost(tile)
    total_cycles = sum(c.cycles for c in mvm_costs + nonmvm_costs)
    total_energy = sum(c.energy_pJ for c in mvm_costs + nonmvm_costs)
    w = Workload("ResNet-20", hcts_per_instance(mvm_costs), total_cycles,
                total_energy, unit="inferences")
    dp = Chip().run(w)
    base = ResNet20Baseline(tile=tile, round_trip_latency_ns=CALIBRATED_ROUND_TRIP_NS,
                            link_GB_per_s=CALIBRATED_LINK_GB_PER_S).run()

    throughput_speedup = dp.throughput_per_s / base.throughput_inferences_per_s
    latency_speedup = base.single_inference_latency_us / (dp.latency_ns / 1000)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    panels = [
        (axes[0], "Throughput speedup", 14.8, throughput_speedup, "{:.2f}×"),
        (axes[1], "Latency speedup", 1.667, latency_speedup, "{:.2f}×"),
    ]
    for ax, title, paper_val, our_val, fmt in panels:
        bars = ax.bar(["Paper", "Ours"], [paper_val, our_val],
                      color=[NEUTRAL, BLUE], width=0.55)
        bar_value_labels(ax, bars, fmt=fmt)
        ax.set_title(title)
        ax.set_ylim(0, max(paper_val, our_val) * 1.25)

    fig.suptitle("ResNet-20 vs. Baseline: paper vs. our reproduction", y=1.04)
    save(fig, "reproduction_summary")
    return throughput_speedup, latency_speedup


# --------------------------------------------------------------------------
# Figure 2: ResNet-20 accuracy under increasing analog non-idealities.
# --------------------------------------------------------------------------

def fig_resnet20_accuracy():
    # Measured on the full 10,000-image CIFAR-10 test set -- see
    # workloads/resnet20/accuracy.py for the run that produced these.
    labels = ["fp32\n(no analog)", "CrossSim\nideal", "With device\nnoise (1 bit/cell)"]
    values = [91.69, 91.69, 83.43]
    colors = [NEUTRAL, BLUE, ORANGE]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars = ax.bar(labels, values, color=colors, width=0.6)
    bar_value_labels(ax, bars, fmt="{:.2f}%")
    ax.set_ylabel("CIFAR-10 top-1 accuracy (%)")
    ax.set_title("ResNet-20 accuracy under analog non-idealities")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=9)
    save(fig, "resnet20_accuracy")


# --------------------------------------------------------------------------
# Figure 3: the precision/throughput tradeoff contribution.
# --------------------------------------------------------------------------

def fig_precision_tradeoff():
    from contribution.precision_tradeoff import throughput_sweep

    points = throughput_sweep()
    base_throughput = points[0].chip_throughput_per_s
    labels = [f"{p.cell_bits} bit{'s' if p.cell_bits > 1 else ''}/cell" for p in points]
    speedups = [p.chip_throughput_per_s / base_throughput for p in points]
    accuracies = [p.accuracy_pct for p in points]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    b1 = axes[0].bar(labels, speedups, color=BLUE_RAMP, width=0.6)
    bar_value_labels(axes[0], b1, fmt="{:.2f}×")
    axes[0].set_title("Chip throughput")
    axes[0].set_ylim(0, max(speedups) * 1.25)

    b2 = axes[1].bar(labels, accuracies, color=BLUE_RAMP, width=0.6)
    bar_value_labels(axes[1], b2, fmt="{:.1f}%")
    axes[1].set_title("CIFAR-10 accuracy")
    axes[1].set_ylim(0, 100)

    fig.suptitle("Bits per analog cell: throughput vs. accuracy", y=1.04)
    save(fig, "precision_tradeoff")


# --------------------------------------------------------------------------
# Figure 4: the zero-skipping contribution -- why the real number is small.
# --------------------------------------------------------------------------

def fig_zero_skip_dilution():
    from darthpum.hct import HybridComputeTile
    from workloads.resnet20.resnet_pum import inference_cost

    tile = HybridComputeTile()
    base_mvm, base_nonmvm = inference_cost(tile, zero_skip=False)
    skip_mvm, skip_nonmvm = inference_cost(tile, zero_skip=True)
    base_energy = sum(c.energy_pJ for c in base_mvm + base_nonmvm)
    skip_energy = sum(c.energy_pJ for c in skip_mvm + skip_nonmvm)
    measured_pct = 100 * (1 - skip_energy / base_energy)
    naive_pct = 48.0  # 51.1% mean sparsity x ~94% isolated Row Periphery share

    labels = ["Naive estimate\n(isolated kernel)", "Measured\n(end-to-end)"]
    values = [naive_pct, measured_pct]
    colors = [ORANGE, GREEN]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars = ax.bar(labels, values, color=colors, width=0.55)
    bar_value_labels(ax, bars, fmt="{:.1f}%")
    ax.set_ylabel("ResNet-20 analog energy reduction")
    ax.set_title("Zero-skipping: why the honest number is small")
    ax.set_ylim(0, 55)
    save(fig, "zero_skip_dilution")
    return measured_pct


# --------------------------------------------------------------------------
# Figure 5: ResNet-20's per-layer array footprint (why throughput comes from
# replication, not per-layer speed).
# --------------------------------------------------------------------------

def fig_layer_footprint():
    from darthpum.hct import HybridComputeTile
    from workloads.resnet20.mapping import map_resnet20

    tile = HybridComputeTile()
    mappings = map_resnet20(ace=tile.ace)
    names = [m.name for m in mappings]
    tiles = [m.tiles for m in mappings]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(range(len(names)), tiles, color=BLUE, width=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Physical arrays (INT8/SLC)")
    ax.set_title(f"ResNet-20: array footprint per layer "
                f"(largest: {max(tiles)}, {names[tiles.index(max(tiles))]})",
                fontsize=10)
    ax.set_ylim(0, max(tiles) * 1.15)
    save(fig, "layer_footprint")
    return sum(tiles)


# --------------------------------------------------------------------------
# Figure 6: OSCAR digital cost table cross-check against RACER's published
# MAGIC-family NOR count.
# --------------------------------------------------------------------------

def fig_oscar_crosscheck():
    from darthpum.recipes import load

    ops = ["XOR", "ADD", "MUL8", "MUL16", "MUL32"]
    recipes = load()
    cycles = [recipes[op].cycles for op in ops]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars = ax.bar(ops, cycles, color=BLUE, width=0.6)
    bar_value_labels(ax, bars, fmt="{:.0f}")
    ax.set_yscale("log")
    ax.set_ylabel("Cycles per operation (log scale)")
    ax.set_title("Parsed OSCAR cost table, by operation")
    save(fig, "oscar_crosscheck")
    return recipes["ADD"].cycles, recipes["ADD"].mix.get("NOR", 0)


# --------------------------------------------------------------------------
# Submission folders: the assignment asks for "Baseline Results" and
# "Contribution Results" as their own graphs/tables, separate from the
# report. These copy the already-generated figures and add a short summary
# table each, so the two deliverables stand on their own.
# --------------------------------------------------------------------------

BASELINE_README = """# Baseline Results

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
"""

CONTRIBUTION_README = """# Contribution Results

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
"""


def export_submission_folders():
    """Copy the relevant figures into results/baseline/ and
    results/contribution/, per the assignment's separate upload fields."""
    os.makedirs("results/baseline", exist_ok=True)
    os.makedirs("results/contribution", exist_ok=True)

    for name in ("reproduction_summary", "resnet20_accuracy",
                "layer_footprint", "oscar_crosscheck"):
        shutil.copy(f"paper/figs/{name}.pdf", f"results/baseline/{name}.pdf")
    with open("results/baseline/README.md", "w") as f:
        f.write(BASELINE_README)

    for name in ("zero_skip_dilution", "precision_tradeoff"):
        shutil.copy(f"paper/figs/{name}.pdf", f"results/contribution/{name}.pdf")
    with open("results/contribution/README.md", "w") as f:
        f.write(CONTRIBUTION_README)

    print("  wrote results/baseline/ and results/contribution/")


if __name__ == "__main__":
    print("Generating figures...")
    throughput_speedup, latency_speedup = fig_reproduction_summary()
    fig_resnet20_accuracy()
    fig_precision_tradeoff()
    measured_pct = fig_zero_skip_dilution()
    total_tiles = fig_layer_footprint()
    add_cycles, add_nor = fig_oscar_crosscheck()
    export_submission_folders()
    print(f"\nThroughput speedup: {throughput_speedup:.2f}x  "
         f"Latency speedup: {latency_speedup:.2f}x  "
         f"Zero-skip reduction: {measured_pct:.2f}%  "
         f"Total physical arrays: {total_tiles}  "
         f"ADD: {add_cycles} cycles ({add_nor} NOR)")
    print("\nDone. See paper/figs/*.pdf, results/baseline/, results/contribution/")
