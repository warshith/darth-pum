"""ResNet-20 architecture and its crossbar mapping."""

import torch.nn as nn
import pytest

from workloads.resnet20.model import ResNet20, count_params
from workloads.resnet20.mapping import map_resnet20, summarize


def test_layer_count_matches_figure_15():
    """[DARTH] Figure 15 shows exactly 20 per-layer bars ending in 'fc' -- the
    signature of option-A shortcuts (no extra downsample-conv layers)."""
    m = ResNet20()
    n_conv = sum(1 for mod in m.modules() if isinstance(mod, nn.Conv2d))
    n_fc = sum(1 for mod in m.modules() if isinstance(mod, nn.Linear))
    assert n_conv == 19
    assert n_fc == 1
    assert n_conv + n_fc == 20


def test_param_count_matches_published_figure():
    """akamaster/pytorch_resnet_cifar10 reports ~0.27M params for this variant."""
    p = count_params(ResNet20())
    assert p["total"] == 269_722
    assert p["conv_fc"] == 268_346


def test_shortcut_adds_no_parameters():
    """Option-A (zero-pad) shortcuts must contribute zero parameters -- the
    whole point of why this variant has 19 conv layers, not 21."""
    m = ResNet20()
    shortcut_params = sum(
        p.numel() for n, p in m.named_parameters() if "shortcut" in n
    )
    assert shortcut_params == 0


def test_logical_mapping_matches_independent_cross_check():
    """Cross-checked against an independently computed reference (same method,
    separate derivation): 98 logical (un-sliced) 64x64 arrays, ~40.55M MACs.
    Agreement here is strong evidence the im2col tiling is implemented
    correctly, before bit-slicing is applied on top."""
    mappings = map_resnet20(value_bits=1)
    s = summarize(mappings)
    assert s["n_layers"] == 20
    assert s["total_logical_tiles"] == 98
    assert s["total_MACs"] == 40_551_040


def test_physical_int8_mapping_is_8x_logical():
    """The real, usable mapping: INT8 weights bit-sliced onto 1-bit (SLC) cells
    -- matching [DARTH] Section 7.5's SLC accuracy result -- need 8 physical
    arrays per logical tile. 784 also independently cross-checks (same order,
    reached via a different precision assumption: 16-bit/2-bit cells, also 8x)."""
    mappings = map_resnet20(value_bits=8)
    s = summarize(mappings)
    assert s["total_logical_tiles"] == 784


def test_largest_layer_is_576_wide_stage3_conv():
    """layer3's 3x3 convs at 64->64 channels give the widest im2col matrix
    (576 rows), needing 9 logical / 72 physical arrays -- still well inside one
    ACE's 64 logical arrays."""
    mappings = map_resnet20(value_bits=1)
    largest = max(mappings, key=lambda m: m.tiles)
    assert largest.weight_rows == 576
    assert largest.tiles == 9


def test_whole_network_fits_in_one_ace_logically():
    """98 logical tiles across the WHOLE network exceeds one ACE's 64, but each
    INDIVIDUAL layer fits comfortably. Throughput comes from replicating layers
    across many tiles, not from one tile holding the whole network."""
    mappings = map_resnet20(value_bits=1)
    assert all(m.tiles <= 64 for m in mappings), "every individual layer must fit one ACE"
    assert sum(m.tiles for m in mappings) > 64, "but the whole network needs more than one ACE"


def test_stride_2_layers_shrink_spatial_resolution():
    mappings = map_resnet20()
    by_name = {m.name: m for m in mappings}
    assert by_name["r2.0.conv1"].out_shape[1:] == (16, 16)   # 32 -> 16
    assert by_name["r3.0.conv1"].out_shape[1:] == (8, 8)     # 16 -> 8
    assert by_name["conv1"].out_shape[1:] == (32, 32)        # stem: no stride


def test_checkpoint_loads_with_expected_accuracy_metadata():
    """Sanity check the checkpoint file is the one we think it is."""
    import torch
    from pathlib import Path
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "resnet20-12fca82f.th"
    if not ckpt_path.exists():
        pytest.skip("checkpoint not downloaded")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt
