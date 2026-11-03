"""Measured ReLU output sparsity for our ResNet-20 checkpoint on real CIFAR-10
images. This is the evidence the zero-skipping ACE contribution rests on.

Everyone knows post-ReLU activations are sparse (negative pre-activation ->
exact zero), but "everyone knows" isn't a number we can put in a table, so
this module measures OUR model's actual sparsity instead of assuming some
textbook rate -- same rule we've tried to follow for every cost figure in
this project. measure() hooks torch.nn.functional.relu directly, since
model.py calls it functionally in forward() rather than as an nn.ReLU module
(a module-level hook would just silently never fire), and logs the fraction
of exact zeros over one real CIFAR-10 test batch.

What we got, 256 test images on the checkpoint we're actually using: mean
51.1% zeros across the network's 19 ReLU sites, from 15.3% on an early layer
up to 81.3% on a late one. MEASURED_SPARSITY below is that result, cached so
nothing downstream has to re-run inference on every import -- call measure()
if you want it fresh.

How much is that sparsity actually worth? See the active_row_fraction note in
darthpum/ace.py. Short version: disabling a zero row's wordline driver only
touches "Row Periphery", a small slice of ACE energy once you account for ADC
conversion (runs regardless) and digital recombination of partial sums (also
runs regardless). End to end on the real network, 51% sparsity buys roughly
3% total energy, not 51% -- smaller than it sounds, but for a reason we can
actually point to.
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
    """Regenerate MEASURED_SPARSITY from scratch -- real inference, real
    images, real zeros counted. Needs torch/torchvision plus the checkpoint
    and CIFAR-10 test set downloaded."""
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

    Layer order from mapping.py: [conv1, r1.0.conv1, r1.0.conv2, r1.1.conv1,
    r1.1.conv2, ..., r3.2.conv2, fc] -- 20 of them. conv1 is the stem, eats
    the raw image (not sparse at all), so it just gets 0.0. The other 19 each
    consume exactly one ReLU's output, in forward-pass order, so
    MEASURED_SPARSITY[i] lines up with layer i+1.
    """
    from .mapping import map_resnet20

    names = [m.name for m in map_resnet20()]
    assert len(names) == len(MEASURED_SPARSITY) + 1, (
        f"expected {len(MEASURED_SPARSITY) + 1} layers, mapping.py gives {len(names)} -- "
        "did the ResNet-20 architecture change? the ReLU-to-layer mapping needs redoing"
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
