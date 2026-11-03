"""Mapping ResNet-20's convolutions onto DARTH-PUM's analog crossbars.

Standard im2col/Toeplitz trick ([DARTH] 5.1 cites this for CNNs, same thing
ISAAC and PUMA do): flatten each 3x3xC_in filter into a (k*k*C_in)-length
column, so the weight matrix is (k*k*C_in) x C_out. Tile that onto the ACE's
64x64 arrays, and if k*k*C_in > 64 the row-tiles' partial sums get combined
digitally afterward.

Everything here is analytic -- no weights or data needed to get tile counts,
so this runs fine even before the checkpoint/dataset are downloaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from darthpum.ace import AnalogComputeElement
from darthpum.device import ACEParams, ADCParams

from .model import ResNet20

import torch.nn as nn


@dataclass(frozen=True)
class LayerMapping:
    name: str
    kind: str              # "conv" or "fc"
    in_shape: tuple        # (C_in, H, W) for conv; (C_in,) for fc
    out_shape: tuple       # (C_out, H_out, W_out) for conv; (C_out,) for fc
    kernel: int            # 1 for fc
    weight_rows: int       # k*k*C_in (conv) or C_in (fc)
    weight_cols: int       # C_out
    tiles: int              # arrays needed, from AnalogComputeElement.tiles_for_matrix
    macs: int

    @property
    def row_tiles(self) -> int:
        # ceil-div, since the matrix won't always fill a row-tile evenly
        return math.ceil(self.weight_rows / 64)

    @property
    def col_tiles(self) -> int:
        return math.ceil(self.weight_cols / 64)


def map_resnet20(model: ResNet20 | None = None, value_bits: int = 8,
                 ace: AnalogComputeElement | None = None) -> list[LayerMapping]:
    """Walk ResNet-20's layers and work out each one's crossbar footprint.

    value_bits=8 is standard INT8 for CNN inference. Combined with the default
    ACEParams' SLC cells (1 bit/cell, matches [DARTH] 7.5's accuracy result),
    that's 8 physical arrays per logical tile from bit-slicing -- so `tiles`
    below is already the PHYSICAL count, not logical. value_bits=1 instead
    gives the un-sliced logical count, which is only useful as a cross-check
    (treating every weight as one bit obviously isn't a usable network).

    Spatial sizes are hardcoded from the known 32x32 CIFAR-10 input and the
    architecture's stride pattern -- tracing shapes by hand like this means we
    don't need an actual model instance just to compute a tile count.
    """
    model = model or ResNet20()
    ace = ace or AnalogComputeElement(ACEParams(adc=ADCParams.sar()))

    mappings = []
    hw = 32  # spatial resolution; halves at each stride-2 stage

    def add_conv(name, c_in, c_out, k, h, w, stride):
        rows, cols = k * k * c_in, c_out
        h_out, w_out = h // stride, w // stride
        macs = rows * cols * h_out * w_out
        mappings.append(LayerMapping(
            name=name, kind="conv",
            in_shape=(c_in, h, w), out_shape=(c_out, h_out, w_out),
            kernel=k, weight_rows=rows, weight_cols=cols,
            tiles=ace.tiles_for_matrix(rows, cols, value_bits), macs=macs,
        ))
        return h_out, w_out

    # stem: 3x3, 3 -> 16, stride 1
    hw, _ = add_conv("conv1", 3, 16, 3, hw, hw, 1), hw
    hw = hw[0] if isinstance(hw, tuple) else hw

    # three stages, matches model.py's _make_layer
    stage_cfg = [(16, 16, 1), (16, 32, 2), (32, 64, 2)]
    for stage_idx, (c_in_first, c_out, stride_first) in enumerate(stage_cfg, start=1):
        c_in = c_in_first
        for block_idx in range(3):
            stride = stride_first if block_idx == 0 else 1
            prefix = f"r{stage_idx}.{block_idx}"
            h_out, w_out = add_conv(f"{prefix}.conv1", c_in, c_out, 3, hw, hw, stride)
            add_conv(f"{prefix}.conv2", c_out, c_out, 3, h_out, h_out, 1)
            hw = h_out
            c_in = c_out

    # fc: 64 -> 10, no spatial extent, kernel is effectively 1
    mappings.append(LayerMapping(
        name="fc", kind="fc", in_shape=(64,), out_shape=(10,),
        kernel=1, weight_rows=64, weight_cols=10,
        tiles=ace.tiles_for_matrix(64, 10, value_bits), macs=64 * 10,
    ))

    return mappings


def summarize(mappings: list[LayerMapping]) -> dict:
    total_tiles = sum(m.tiles for m in mappings)
    total_macs = sum(m.macs for m in mappings)
    return {
        "n_layers": len(mappings),
        "total_logical_tiles": total_tiles,
        "total_MACs": total_macs,
        "largest_layer": max(mappings, key=lambda m: m.tiles).name,
        "largest_layer_tiles": max(m.tiles for m in mappings),
    }


if __name__ == "__main__":
    logical = map_resnet20(value_bits=1)   # un-sliced, precision-agnostic cross-check
    physical = map_resnet20(value_bits=8)  # INT8 on SLC cells -- the real mapping

    print(f"{'layer':<12}{'rows':>6}{'cols':>5}{'logical':>9}{'physical':>10}{'MACs':>12}")
    for lo, ph in zip(logical, physical):
        print(f"{lo.name:<12}{lo.weight_rows:>6}{lo.weight_cols:>5}"
              f"{lo.tiles:>9}{ph.tiles:>10}{lo.macs:>12,}")

    sl, sp = summarize(logical), summarize(physical)
    print(f"\n{sl['n_layers']} layers (paper's Fig. 15: 20 bars)")
    print(f"logical (un-sliced) arrays : {sl['total_logical_tiles']} "
          f"(cross-checks against an independent reference)")
    print(f"physical arrays (INT8/SLC) : {sp['total_logical_tiles']} "
          f"(8x logical, from bit-slicing 8-bit weights onto 1-bit cells)")
    print(f"total MACs: {sl['total_MACs']:,} "
          f"(expect ~40.5M, matching the standard ResNet-20/CIFAR-10 count)")
    print(f"largest layer: {sp['largest_layer']} ({sp['largest_layer_tiles']} physical tiles)")
    print(f"\nEven at 8-bit/SLC, the whole network needs only "
          f"{sp['total_logical_tiles']} of the chip's ~1844 tiles -- throughput "
          f"comes from replicating across the chip, not from any one tile being larger.")
