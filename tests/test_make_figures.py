"""Figure generation: catches the kind of mistake that matters most here --
plotting a number that doesn't match what the investigation actually reported.
"""

import matplotlib
matplotlib.use("Agg")  # no display needed for tests

import pytest


def test_resnet20_speedup_matches_the_investigated_value():
    from experiments.make_figures import fig_reproduction_summary
    throughput_speedup, latency_speedup = fig_reproduction_summary()
    assert throughput_speedup == pytest.approx(14.72, rel=0.02)
    assert latency_speedup == pytest.approx(1.67, rel=0.02)


def test_zero_skip_figure_matches_the_investigated_value():
    from experiments.make_figures import fig_zero_skip_dilution
    measured_pct = fig_zero_skip_dilution()
    assert measured_pct == pytest.approx(3.07, rel=0.02)


def test_layer_footprint_matches_the_known_total():
    from experiments.make_figures import fig_layer_footprint
    total_tiles = fig_layer_footprint()
    assert total_tiles == 784


def test_oscar_crosscheck_matches_the_parsed_add_recipe():
    from experiments.make_figures import fig_oscar_crosscheck
    add_cycles, add_nor = fig_oscar_crosscheck()
    assert add_cycles == 21
    assert add_nor == 16


def test_all_figures_are_written(tmp_path, monkeypatch):
    """A smoke test that every figure function runs end-to-end and writes a
    PDF for the report."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "paper" / "figs").mkdir(parents=True)

    import importlib
    import experiments.make_figures as mf
    importlib.reload(mf)  # re-run os.makedirs under the new cwd

    mf.fig_reproduction_summary()
    mf.fig_resnet20_accuracy()
    mf.fig_precision_tradeoff()
    mf.fig_zero_skip_dilution()
    mf.fig_layer_footprint()
    mf.fig_oscar_crosscheck()

    for name in ("reproduction_summary", "resnet20_accuracy",
                "precision_tradeoff", "zero_skip_dilution",
                "layer_footprint", "oscar_crosscheck"):
        assert (tmp_path / "paper" / "figs" / f"{name}.pdf").exists()


def test_submission_folders_get_the_right_figures(tmp_path, monkeypatch):
    """results/baseline/ and results/contribution/ are their own deliverable
    per the assignment brief, so each must hold the figures that back its
    claims, not just whatever happened to be generated."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "paper" / "figs").mkdir(parents=True)

    import importlib
    import experiments.make_figures as mf
    importlib.reload(mf)

    mf.fig_reproduction_summary()
    mf.fig_resnet20_accuracy()
    mf.fig_precision_tradeoff()
    mf.fig_zero_skip_dilution()
    mf.fig_layer_footprint()
    mf.fig_oscar_crosscheck()
    mf.export_submission_folders()

    for name in ("reproduction_summary", "resnet20_accuracy",
                "layer_footprint", "oscar_crosscheck"):
        assert (tmp_path / "results" / "baseline" / f"{name}.pdf").exists()
    assert (tmp_path / "results" / "baseline" / "README.md").exists()

    for name in ("zero_skip_dilution", "precision_tradeoff"):
        assert (tmp_path / "results" / "contribution" / f"{name}.pdf").exists()
    assert (tmp_path / "results" / "contribution" / "README.md").exists()
