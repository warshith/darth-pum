"""Measured ReLU output sparsity for our ResNet-20 checkpoint on real CIFAR-10
images -- the evidence base for the zero-skipping ACE contribution.

CNN activations are widely known to be sparse after ReLU (any negative
pre-activation becomes exactly zero), but "widely known" is not evidence:
this module measures our OWN model's actual sparsity rather than assume a
textbook figure, the same discipline as every cost number elsewhere in this
project. `measure()` hooks `torch.nn.functional.relu` (the model calls it
functionally in model.py's forward(), not as an nn.ReLU module, so a module
hook would silently fire zero times) and records the fraction of exact zeros
in its output, over one real batch of CIFAR-10 test images.

Result, 256 test images, our verified checkpoint: mean 51.1% zero activations
across the network's 19 ReLU sites, ranging from 15.3% (an early layer, few
dead units yet) to 81.3% (a late layer). The MEASURED_SPARSITY tuple below is
this measurement, stored so the contribution's numbers do not require
re-running inference on every import; `measure()` regenerates it.

What this sparsity is actually worth: see darthpum/ace.py's
`active_row_fraction` docstring. Disabling a zero row's wordline driver only
touches the "Row Periphery" energy term, which is a small fraction of total
ACE energy once ADC conversion (always active, unaffected by which rows were
zero) and digital recombination of bit-sliced partial sums (also unaffected)
are accounted for. Measured end-to-end on the whole ResNet-20 network, this
51% sparsity yields roughly a 3% total energy reduction -- real, but much
smaller than the 51% figure alone would suggest, for a specific, identified
reason rather than an unexplained shortfall.
"""

from __future__ import annotations

from pathlib import Path

CHECKPOINT = str(Path(__file__).resolve().parent.parent.parent
                 / "checkpoints" / "resnet20-12fca82f.th")
DATA_ROOT = str(Path(__file__).resolve().parent.parent.parent / "data")

# Measured, in forward-pass order: stem-ReLU, then each BasicBlock's internal
# ReLU (after conv1) and output ReLU (after the residual add), for all 9
# blocks across the three stages. 19 values, one per ReLU site.
MEASURED_SPARSITY: tuple[float, ...] = (
    0.364, 0.504, 0.311, 0.524, 0.244, 0.615, 0.153, 0.550, 0.327,
    0.749, 0.328, 0.799, 0.345, 0.642, 0.546, 0.781, 0.606, 0.813, 0.502,
)

MEASUREMENT_BATCH_SIZE = 256
MEASUREMENT_MEAN = sum(MEASURED_SPARSITY) / len(MEASURED_SPARSITY)


def measure(n_images: int = 256) -> tuple[float, ...]:
    """Regenerate MEASURED_SPARSITY from scratch: real inference, real images,
    real zeros counted -- not estimated. Requires torch/torchvision and the
    downloaded checkpoint + CIFAR-10 test set."""
    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms

    from .model import load_pretrained

    model = load_pretrained(CHECKPOINT)
    model.eval()

    sparsity = []
    orig_relu = F.relu

    def traced_relu(x, *a, **kw):
        out = orig_relu(x, *a, **kw)
        sparsity.append((out == 0).float().mean().item())
        return out

    F.relu = traced_relu
    try:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        ds = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=tf)
        loader = torch.utils.data.DataLoader(ds, batch_size=n_images, shuffle=False)
        x, _ = next(iter(loader))
        with torch.no_grad():
            model(x)
    finally:
        F.relu = orig_relu

    return tuple(sparsity)


def layer_skip_fractions() -> dict[str, float]:
    """Map the 19 measured ReLU sparsities onto mapping.py's 20 layer names.

    mapping.py's layer order is [conv1, r1.0.conv1, r1.0.conv2, r1.1.conv1,
    r1.1.conv2, ..., r3.2.conv2, fc] -- 20 entries. conv1 (the stem) consumes
    the raw image, which is not sparse, so it gets 0.0 (no skipping). Each of
    the remaining 19 layers consumes the output of exactly one ReLU, in the
    same order the forward pass executes them, so MEASURED_SPARSITY[i] is
    layer i+1's input skip fraction.
    """
    from .mapping import map_resnet20

    names = [m.name for m in map_resnet20()]
    assert len(names) == len(MEASURED_SPARSITY) + 1, (
        f"expected {len(MEASURED_SPARSITY) + 1} layers, mapping.py gives {len(names)} "
        "-- the ResNet-20 architecture changed; re-derive the ReLU-to-layer mapping"
    )
    out = {names[0]: 0.0}
    for name, s in zip(names[1:], MEASURED_SPARSITY):
        out[name] = s
    return out


if __name__ == "__main__":
    fresh = measure()
    print(f"Measured {len(fresh)} ReLU sites, {MEASUREMENT_BATCH_SIZE} images:")
    for i, s in enumerate(fresh):
        print(f"  site {i:2d}: {100*s:.1f}%")
    print(f"\nmean: {100*sum(fresh)/len(fresh):.1f}%  "
          f"(stored MEASURED_SPARSITY mean: {100*MEASUREMENT_MEAN:.1f}%)")

    print("\nPer-layer skip fractions:")
    for name, frac in layer_skip_fractions().items():
        print(f"  {name:<12}{100*frac:5.1f}%")
