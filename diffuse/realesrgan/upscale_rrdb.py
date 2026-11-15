#!/usr/bin/env python3
"""Real-ESRGAN x4 (RRDBNet) standalone: carrega weights da casa e faz upscale tileado.

Sem basicsr/gfpgan: arch RRDBNet implementada inline em torch puro,
state dict compativel com os keys do RealESRGAN_x4plus.pth (basicsr RRDBNet).

Uso:  python upscale_rrdb.py <in.(png|jpg)> <out.png> [--scale 4] [--tile 384]
  --scale: outscale final (padrao 4; use 2 para fechar em metade do ganho)
"""
import argparse
import functools
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, **kwargs):
        super().__init__()
        num_pad = 2 if num_in_ch % 2 == 0 else 1
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def model_path():
    cands = [
        os.path.join(ROOT, "weights", "RealESRGAN_x4plus.pth"),
        os.path.expanduser("~/git/ai-dotfiles/diffuse/realesrgan/weights/RealESRGAN_x4plus.pth"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("RealESRGAN_x4plus.pth nao encontrado: " + repr(cands))


def load_model(device):
    net = RRDBNet()
    state = torch.load(model_path(), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "params_ema" in state:
        state = state["params_ema"]
    if isinstance(state, dict) and "params" in state:
        state = state["params"]
    net.load_state_dict(state, strict=True)
    net.eval().to(device)
    if device == "cuda":
        net.half()
    return net


@torch.inference_mode()
def upscale_tiled(net, in_tensor, tile=384, pad=10, progress=None):
    """in_tensor: 1x3xHxW (float, 0-1, metade se half no device). tile em px da ENTRADA."""
    if tile is None or tile <= 0:
        out = net(in_tensor)
        return out.clamp_(0, 1)
    _, _, h, w = in_tensor.shape
    out = torch.empty((1, 3, h * 4, w * 4), dtype=in_tensor.dtype, device=in_tensor.device)
    cols = math.ceil(w / tile)
    rows = math.ceil(h / tile)
    total = cols * rows
    done = 0
    for r in range(rows):
        for c in range(cols):
            y0 = r * tile
            x0 = c * tile
            y1 = min(h, y0 + tile)
            x1 = min(w, x0 + tile)
            ys, ye = max(0, y0 - pad), min(h, y1 + pad)
            xs, xe = max(0, x0 - pad), min(w, x1 + pad)
            patch = in_tensor[:, :, ys:ye, xs:xe]
            up = net(patch).clamp_(0, 1)
            uy0, ux0 = (y0 - ys) * 4, (x0 - xs) * 4
            out[:, :, y0 * 4:y1 * 4, x0 * 4:x1 * 4] = up[:, :, uy0:uy0 + (y1 - y0) * 4, ux0:ux0 + (x1 - x0) * 4]
            done += 1
            if progress:
                print(f"  tile {done}/{total}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--scale", type=float, default=4.0, help="scale final (default 4; 2 = metade do ganho)")
    ap.add_argument("--tile", type=int, default=384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = Image.open(args.input).convert("RGB")
    w, h = img.size
    print(f"in: {w}x{h} | outscale {args.scale} | tile {args.tile} | device {device}", flush=True)
    tensor = torch.from_numpy(__import__("numpy").asarray(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    if device == "cuda":
        tensor = tensor.half()
    net = load_model(device)
    out = upscale_tiled(net, tensor, tile=args.tile)
    out = out.float().clamp(0, 1)
    if args.scale != 4.0:
        out = F.interpolate(out, scale_factor=args.scale / 4.0, mode="bicubic", align_corners=False).clamp(0, 1)
    arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr).save(args.output, "PNG")
    ow, oh = arr.shape[1], arr.shape[0]
    print(f"out: {ow}x{oh} -> {args.output}")


if __name__ == "__main__":
    main()
