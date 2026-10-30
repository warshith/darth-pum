"""ResNet-20 accuracy under CrossSim: config wiring and a fast smoke test.

The full 10,000-image evaluation takes minutes, too slow for a unit test suite
-- these tests use tiny image counts (fast) to catch wiring bugs (the kind
that crashed the first run: drift_model defaulting to Python None instead of
the string "none"), not to validate the accuracy NUMBER itself.
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
    """The bug this guards against: dnn_inference_params defaults
    drift_model to Python None, not the string 'none', which crashes
    CrossSim's device factory unless set explicitly. Caught on the first
    real run before any accuracy number was trusted."""
    from workloads.resnet20.accuracy import _device_params
    p = _device_params(ideal=True)
    assert p is not None


def test_noisy_params_build_without_crashing():
    from workloads.resnet20.accuracy import _device_params
    p = _device_params(ideal=False, error_model="RRAMMilo", adc_type="sar")
    assert p is not None


def test_ideal_model_matches_fp32_on_a_tiny_sample():
    """With no non-idealities, CrossSim's conversion must reproduce the plain
    PyTorch model's predictions exactly -- this is the sanity check that
    catches conversion bugs (wrong layer mapping, bad BN fusion) independently
    of anything noise-related."""
    from workloads.resnet20.accuracy import build_analog_model, evaluate
    from workloads.resnet20.model import load_pretrained

    fp32 = load_pretrained(str(CHECKPOINT))
    fp32_acc = evaluate(fp32, n_images=20, batch_size=20)

    ideal = build_analog_model(ideal=True)
    ideal_acc = evaluate(ideal, n_images=20, batch_size=20)

    assert ideal_acc == pytest.approx(fp32_acc)


def test_noisy_model_runs_and_gives_a_valid_accuracy():
    """Not a check against the paper's 75.4% (20 images is far too small a
    sample for that) -- just confirms the full RRAMMilo + SLC + SAR path
    executes end-to-end and returns a value in a sane range."""
    from workloads.resnet20.accuracy import build_analog_model, evaluate

    noisy = build_analog_model(ideal=False)
    acc = evaluate(noisy, n_images=20, batch_size=20)
    assert 0.0 <= acc <= 100.0


def test_noisy_accuracy_is_not_better_than_ideal():
    """Device noise should not somehow IMPROVE accuracy over the noise-free
    case -- a basic direction check, even on a tiny, noisy-by-sample-size
    comparison (allowing some slack since 20 images is a small sample)."""
    from workloads.resnet20.accuracy import build_analog_model, evaluate

    ideal_acc = evaluate(build_analog_model(ideal=True), n_images=200, batch_size=100)
    noisy_acc = evaluate(build_analog_model(ideal=False), n_images=200, batch_size=100)
    assert noisy_acc <= ideal_acc + 5.0  # small slack for sampling noise
