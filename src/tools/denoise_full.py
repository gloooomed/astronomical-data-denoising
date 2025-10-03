# src/tools/denoise_full.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

from models.unet_blindspot import UNetBlindspot
from utils.stitch import stitch_tiles


def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_csv", type=str, required=True)   # patches_50_splits.csv
    ap.add_argument("--parent_id", type=str, required=True)   # e.g., fd2582f57237
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)         # output PNG
    args = ap.parse_args()

    df = pd.read_csv(args.patch_csv)
    sub = df[df["parent_id"] == args.parent_id].copy()
    if len(sub) == 0:
        raise SystemExit(f"No tiles for parent_id={args.parent_id}")

    # infer full canvas size
    h = int(sub["h"].iloc[0])
    w = int(sub["w"].iloc[0])
    y_max = int(sub["y"].max())
    x_max = int(sub["x"].max())
    H = y_max + h
    W = x_max + w

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetBlindspot(in_ch=3, base=48).to(device).eval()
    load_ckpt(model, args.ckpt)

    tiles, coords = [], []
    with torch.no_grad():
        for row in sub.itertuples():
            arr = np.load(row.path).astype("float32")  # (h,w,C)
            x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)
            y = model(x).clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
            tiles.append(y)
            coords.append((int(row.y), int(row.x)))

    out = stitch_tiles(tiles, coords, out_shape=(H, W, tiles[0].shape[2]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((out*255).astype("uint8")).save(args.out)
    print(f"[info] wrote {args.out}")
