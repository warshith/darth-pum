"""ResNet-20 accuracy under DARTH-PUM's analog non-idealities, via CrossSim.

[DARTH] 7.5 reports 75.4% end-to-end for ResNet-20/CIFAR-10 in SLC (1
bit/cell) mode, vs ~91.7% fp32. This reproduces that comparison with CrossSim
(same simulator the paper uses) on our own verified model/checkpoint, NOT
CrossSim's bundled example ResNet-20. That example uses option-B shortcuts
(1x1 conv downsample), a different architecture from the option-A variant
that actually matches Figure 15's 20-bar layout and that everything else in
this project is built on -- mixing the two would make this accuracy number
inconsistent with the throughput/latency numbers computed elsewhere.

Config, matched to the rest of the project:
  - weight_bits=8, Nslices=8 -> cell_bits=1 (SLC), same as VALUE_BITS/CELL_BITS
    everywhere else in resnet_pum.py and the mapping code.
  - error_model="RRAMMilo" -> the device noise model [DARTH] Sec 6 cites
    (Milo et al., IRPS 2021), already in CrossSim.
  - adc_type="sar" -> matches the paper's default everywhere except the
    explicit SAR-vs-ramp comparison in 7.3.

One real limitation worth flagging: proper input/ADC range calibration (the
per-layer .npy files CrossSim's own example ships with) only exists for
CrossSim's bundled model, not ours, and generating it properly needs a
profiling pass we didn't have time for. So input/ADC quantisation is left at
full precision here (input_bits=0, adc_bits=0), which isolates the thing we
actually care about measuring -- ReRAM weight-storage programming noise, the
source of error the paper's 75.4% is really about -- from quantisation noise
we aren't modelling. Net effect: our noisy accuracy probably comes out a bit
HIGHER than 75.4% since we're modelling less total noise, which is the
direction you'd expect given this, not a sign something's wrong.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "third_party" / "cross-sim"))
sys.path.insert(0, str(_REPO / "third_party" / "cross-sim" / "applications" / "dnn"))

from workloads.resnet20.model import load_pretrained  # noqa: E402

warnings.filterwarnings("ignore")

CHECKPOINT = str(_REPO / "checkpoints" / "resnet20-12fca82f.th")
DATA_ROOT = str(_REPO / "data")

# CIFAR-10 normalisation used by akamaster/pytorch_resnet_cifar10's training
# recipe -- must match what resnet20-12fca82f.th was trained against.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _device_params(weight_bits: int = 8, Nslices: int = 8,
                   error_model: str = "RRAMMilo", adc_type: str = "sar",
                   ideal: bool = False):
    from dnn_inference_params import dnn_inference_params

    return dnn_inference_params(
        ideal=ideal,
        core_style="BALANCED",       # differential cell pairs, [DARTH] Section 2.2.1
        Nslices=Nslices,
        weight_bits=weight_bits,
        digital_bias=True,
        Rmin=1 / 225e-6,             # Gmax=225 uS, RRAMMilo's suggested ceiling
        Rmax=1 / 50e-6,              # Gmin=50 uS, RRAMMilo's suggested floor
        infinite_on_off_ratio=False,
        error_model=error_model,
        drift_model="none",          # dnn_inference_params defaults this to
                                      # Python None, not the string "none", so
                                      # it tries to enable drift with an
                                      # invalid model name unless set explicitly
        Vread=0.5,                   # RRAMMilo's valid I-V range
        # Quantisation left ideal -- see module docstring's scope limitation.
        input_bits=0,
        adc_bits=0,
        adc_type=adc_type,
        useGPU=False,
    )


def build_analog_model(ideal: bool = False):
    """Our verified option-A ResNet-20 with conv/fc swapped for CrossSim
    analog layers. BN gets folded into the preceding conv weights (standard
    inference-time trick) since a real analog array has no separate place to
    stash a BN scale/shift -- it has to be baked into the stored conductances."""
    from simulator.algorithms.dnn.torch.convert import convertible_modules, from_torch

    model = load_pretrained(CHECKPOINT)
    n_layers = len(convertible_modules(model))
    params = [_device_params(ideal=ideal) for _ in range(n_layers)]
    analog = from_torch(model, params, fuse_batchnorm=True, bias_rows=0)
    analog.eval()
    return analog


def cifar10_test_loader(n_images: int, batch_size: int = 100):
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=tf)
    ds = torch.utils.data.Subset(ds, np.arange(min(n_images, len(ds))))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size)


def evaluate(model, n_images: int, batch_size: int = 100, print_progress: bool = False) -> float:
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(cifar10_test_loader(n_images, batch_size)):
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if print_progress:
                print(f"  [{total}/{n_images}] running accuracy: {100*correct/total:.2f}%")
    return 100 * correct / total


if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000, help="number of test images")
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()

    print(f"=== fp32 baseline (no CrossSim) ===")
    fp32_model = load_pretrained(CHECKPOINT)
    t0 = time.time()
    fp32_acc = evaluate(fp32_model, args.n, args.batch, args.progress)
    print(f"fp32 accuracy: {fp32_acc:.2f}%  (akamaster reports 91.73%)  "
          f"[{time.time()-t0:.1f}s]\n")

    print(f"=== CrossSim, ideal (sanity check: should match fp32 closely) ===")
    ideal_model = build_analog_model(ideal=True)
    t0 = time.time()
    ideal_acc = evaluate(ideal_model, args.n, args.batch, args.progress)
    print(f"CrossSim ideal accuracy: {ideal_acc:.2f}%  [{time.time()-t0:.1f}s]\n")

    print(f"=== CrossSim + RRAMMilo noise, SLC (1 bit/cell), SAR ADC ===")
    noisy_model = build_analog_model(ideal=False)
    t0 = time.time()
    noisy_acc = evaluate(noisy_model, args.n, args.batch, args.progress)
    print(f"noisy SLC accuracy: {noisy_acc:.2f}%  (paper: 75.4%)  "
          f"[{time.time()-t0:.1f}s]")
