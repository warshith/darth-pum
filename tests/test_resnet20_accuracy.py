"""CrossSim config wiring + a fast smoke test, not accuracy validation.

The real 10k-image run takes minutes, way too slow to put in a test suite, so
these just use tiny image counts to catch wiring bugs (the drift_model=None
crash below being exhibit A) instead of checking the actual accuracy number.
"""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "resnet20-12fca82f.th"
CIFAR_DIR = Path(__file__).resolve().parent.parent / "data" / "cifar-10-batches-py"

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT.exists() and CIFAR_DIR.exists()),
    reason="checkpoint or CIFAR-10 data not downloaded",
)


def test_ideal_params_build_without_crashing():
    """dnn_inference_params defaults drift_model to None instead of the
    string "none", which crashes CrossSim's device factory if you don't set
    it explicitly -- found this the hard way on the first real run."""
    from workloads.resnet20.accuracy import _device_params
    p = _device_params(ideal=True)
    assert p is not None


def test_noisy_params_build_without_crashing():
    from workloads.resnet20.accuracy import _device_params
    p = _device_params(ideal=False, error_model="RRAMMilo", adc_type="sar")
    assert p is not None


def test_ideal_model_matches_fp32_on_a_tiny_sample():
    """No noise turned on, so CrossSim's conversion should match plain
    PyTorch exactly -- catches conversion bugs (bad layer mapping, broken BN
    fusion) without any noise-related confound."""
    from workloads.resnet20.accuracy import build_analog_model, evaluate
    from workloads.resnet20.model import load_pretrained

    fp32 = load_pretrained(str(CHECKPOINT))
    fp32_acc = evaluate(fp32, n_images=20, batch_size=20)

    ideal = build_analog_model(ideal=True)
    ideal_acc = evaluate(ideal, n_images=20, batch_size=20)

    assert ideal_acc == pytest.approx(fp32_acc)


def test_noisy_model_runs_and_gives_a_valid_accuracy():
    """20 images is nowhere near enough to check against the paper's 75.4%,
    this just confirms the RRAMMilo + SLC + SAR path runs end to end and
    spits out something sane."""
    from workloads.resnet20.accuracy import build_analog_model, evaluate

    noisy = build_analog_model(ideal=False)
    acc = evaluate(noisy, n_images=20, batch_size=20)
    assert 0.0 <= acc <= 100.0


def test_noisy_accuracy_is_not_better_than_ideal():
    """Basic direction check -- noise shouldn't make things BETTER. 200
    images is still small so leave some slack for sampling noise."""
    from workloads.resnet20.accuracy import build_analog_model, evaluate

    ideal_acc = evaluate(build_analog_model(ideal=True), n_images=200, batch_size=100)
    noisy_acc = evaluate(build_analog_model(ideal=False), n_images=200, batch_size=100)
    assert noisy_acc <= ideal_acc + 5.0  # slack for sampling noise
