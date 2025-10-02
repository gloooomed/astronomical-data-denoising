# src/dataio/tiler.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def robust_normalize(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> Tuple[np.ndarray, dict]:
    """Percentile-based normalization to [0,1], protecting faint features."""
    assert img.ndim in (2, 3)
    if img.ndim == 3 and img.shape[2] == 3:
        # Compute percentiles on luminance-ish proxy to avoid color bias
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        lo, hi = np.percentile(gray, [p_low, p_high])
    else:
        lo, hi = np.percentile(img, [p_low, p_high])

    if hi <= lo:
        lo, hi = float(np.min(img)), float(np.max(img))
        if hi == lo:
            hi = lo + 1e-6

    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return img, {"lo": float(lo), "hi": float(hi), "p_low": p_low, "p_high": p_high}


def reflect_pad_to_min(img: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    """Pad with reflection so both dimensions are at least (min_h, min_w)."""
    h, w = img.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return img

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if img.ndim == 2:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    return np.pad(img, pad_width, mode="reflect")


def tile_image(img: np.ndarray, tile: int = 512, stride: int = 256) -> List[Tuple[int, int, np.ndarray]]:
    """Return list of (y, x, tile_array). Handles edge tiles by sticking to the end."""
    H, W = img.shape[:2]
    tiles = []
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))

    # Ensure coverage of the far edge
    if ys[-1] != H - tile:
        ys.append(max(H - tile, 0))
    if xs[-1] != W - tile:
        xs.append(max(W - tile, 0))

    for y in ys:
        for x in xs:
            patch = img[y:y + tile, x:x + tile]
            if patch.shape[0] != tile or patch.shape[1] != tile:
                # Safety: reflect-pad tiny residuals (should be rare after coverage fix)
                patch = reflect_pad_to_min(patch, tile, tile)
            tiles.append((y, x, patch))
    return tiles


def load_image_to_array(path: Path) -> np.ndarray:
    """Read image as float32 in [0,1]; drop alpha if present."""
    with Image.open(path) as im:
        if im.mode == "RGBA":
            im = im.convert("RGB")
        elif im.mode not in ("RGB", "L"):
            # Convert uncommon modes to RGB for consistency
            im = im.convert("RGB")
        arr = np.asarray(im).astype(np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
        elif arr.shape[2] == 4:
            arr = arr[..., :3]
        # Scale 0..255 → 0..1 if needed
        if arr.max() > 1.0:
            arr /= 255.0
        return arr


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Tile images into 512x512 patches with overlap.")
    ap.add_argument("--manifest", type=Path, required=True, help="images.csv created by manifest.py")
    ap.add_argument("--outdir", type=Path, required=True, help="directory to write patch .npy files")
    ap.add_argument("--patch_manifest", type=Path, default=None, help="output CSV path for patch metadata")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--p_low", type=float, default=1.0)
    ap.add_argument("--p_high", type=float, default=99.5)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    ensure_dir(args.outdir)
    if args.patch_manifest is None:
        patch_manifest_path = args.outdir.parent / "manifests" / "patches.csv"
    else:
        patch_manifest_path = args.patch_manifest
    ensure_dir(patch_manifest_path.parent)

    rows = []
    total_tiles = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Tiling"):
        parent_id = r["id"]
        path = Path(r["path"])
        fmt = r["fmt"]
        mode = r["mode"]

        try:
            img = load_image_to_array(path)              # (H, W, C) float32 in [0,1]
            img, norm_stats = robust_normalize(img, args.p_low, args.p_high)
            img = reflect_pad_to_min(img, args.tile, args.tile)

            for y, x, patch in tile_image(img, tile=args.tile, stride=args.stride):
                # Save patch
                patch_fn = f"{parent_id}_y{y}_x{x}.npy"
                patch_path = args.outdir / patch_fn
                np.save(patch_path, patch.astype(np.float32), allow_pickle=False)

                rows.append({
                    "tile_id": f"{parent_id}_{y}_{x}",
                    "parent_id": parent_id,
                    "path": str(patch_path.resolve()),
                    "y": int(y),
                    "x": int(x),
                    "h": int(patch.shape[0]),
                    "w": int(patch.shape[1]),
                    "c": int(patch.shape[2]),
                    "src_fmt": fmt,
                    "src_mode": mode,
                    "norm_lo": norm_stats["lo"],
                    "norm_hi": norm_stats["hi"],
                })
                total_tiles += 1

        except Exception as e:
            print(f"[warn] Skipping {path} due to error: {e}")

    patch_df = pd.DataFrame(rows)
    patch_df.to_csv(patch_manifest_path, index=False)

    print(f"[info] Wrote {total_tiles} patches to: {args.outdir}")
    print(f"[info] Patch manifest: {patch_manifest_path} ({len(patch_df)} rows)")
    if len(patch_df):
        print(f"[info] Example tile: {patch_df.iloc[0]['path']}")
        print(f"[info] Channels distribution: {patch_df['c'].value_counts().to_dict()}")
        print(f"[info] Tile size: {args.tile}  Stride: {args.stride}")


if __name__ == "__main__":
    main()
