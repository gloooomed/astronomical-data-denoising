from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image

from models.unet_blindspot import UNetBlindspot

def load_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_csv", type=str, default="data/manifests/patches_50_splits.csv")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="reports/preview")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.patch_csv)
    df = df[df["split"] == "test"].sample(args.n, random_state=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetBlindspot(in_ch=3, base=48).to(device).eval()
    load_ckpt(model, args.ckpt)

    with torch.no_grad():
        for i, row in enumerate(df.itertuples(), 1):
            arr = np.load(row.path).astype("float32")  # H,W,C in [0,1]
            x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)
            y = model(x).clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()

            Image.fromarray((arr*255).astype("uint8")).save(f"{args.outdir}/in_{i}.png")
            Image.fromarray((y*255).astype("uint8")).save(f"{args.outdir}/out_{i}.png")

    print(f"[info] wrote previews to {args.outdir}")
