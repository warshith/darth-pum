"""ResNet-20 for CIFAR-10, matching the paper-faithful (option-A) architecture.

[DARTH] Figure 15 shows exactly 20 per-layer bars ending in `fc`, with no
downsample-conv bars for the two stride-2 transitions -- that is the signature of
He et al.'s original **option-A** shortcut (parameter-free zero-padding), not the
1x1-conv downsample some later reimplementations use (option-B adds two extra
conv layers, which would show up as two more bars). We therefore match
`akamaster/pytorch_resnet_cifar10`, not `chenyaofo/pytorch-cifar-models` (which is
option-B and ~0.9% more accurate -- a gap larger than most of the quantisation
losses we'll be measuring, so getting the right variant matters).

Architecture: a 3x3 stem (3->16), then three stages of 3 BasicBlocks each
(16/32/64 channels at 32x32/16x16/8x8 resolution), global average pool, and a
64->10 FC layer. 20 weight layers total (19 conv + 1 fc) -- matching the paper's
Figure 15 bar count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/akamaster/pytorch_resnet_cifar10/"
    "master/pretrained_models/resnet20-12fca82f.th"
)
REPORTED_TOP1_ACCURACY = 91.73  # akamaster's reported accuracy for this checkpoint


class LambdaLayer(nn.Module):
    """Option-A shortcut: zero-pad channels, stride-2 subsample spatially.

    No parameters, no extra conv layer -- this is precisely why option-A nets
    have exactly 19 conv layers while option-B nets have 21 (two 1x1 downsample
    convs). Matching the paper's Figure 15 bar count requires option-A.
    """

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # Option A: pad extra channels with zeros, subsample spatially by
            # striding. [DARTH] Figure 15's "r2.0.conv1"-style bars with no
            # matching downsample-conv bar is the tell that this is the variant.
            pad = planes - in_planes
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, pad // 2, pad - pad // 2))
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """19 conv layers (stem + 3 stages x 3 blocks x 2 conv) + 1 FC = 20 total."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        return self.fc(out)


def load_pretrained(checkpoint_path: str) -> ResNet20:
    """Load akamaster's checkpoint, which wraps the model in nn.DataParallel
    (so state dict keys are prefixed with "module.") and stores extra training
    metadata alongside the weights."""
    model = ResNet20()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = {
        k.replace("module.", "", 1).replace("linear.", "fc.", 1): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def count_params(model: nn.Module) -> dict:
    conv_fc = sum(p.numel() for n, p in model.named_parameters() if "bn" not in n)
    bn = sum(p.numel() for n, p in model.named_parameters() if "bn" in n)
    return {"conv_fc": conv_fc, "bn": bn, "total": conv_fc + bn}


if __name__ == "__main__":
    import sys

    m = ResNet20()
    n_conv = sum(1 for mod in m.modules() if isinstance(mod, nn.Conv2d))
    n_fc = sum(1 for mod in m.modules() if isinstance(mod, nn.Linear))
    print(f"ResNet-20: {n_conv} conv + {n_fc} fc = {n_conv + n_fc} weight layers "
          f"(paper's Figure 15: 20 bars)")
    print(count_params(m))

    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(f"forward ok: input {tuple(x.shape)} -> output {tuple(y.shape)}")

    if len(sys.argv) > 1:
        m2 = load_pretrained(sys.argv[1])
        print(f"loaded pretrained checkpoint from {sys.argv[1]}")
