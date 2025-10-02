from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

def make_image_splits(images_df: pd.DataFrame, seed: int, ratios=(0.7, 0.15, 0.15)) -> Dict[str, list]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    ids = images_df["id"].unique().tolist()
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return {"train": train_ids, "val": val_ids, "test": test_ids}

def tag_patches_with_split(patches_df: pd.DataFrame, splits: Dict[str, list]) -> pd.DataFrame:
    id2split = {}
    for split_name, id_list in splits.items():
        for pid in id_list:
            id2split[pid] = split_name
    patches_df = patches_df.copy()
    patches_df["split"] = patches_df["parent_id"].map(id2split).fillna("unassigned")
    return patches_df

def main():
    ap = argparse.ArgumentParser(description="Create parent-level train/val/test splits and tag patch manifest.")
    ap.add_argument("--images", type=Path, required=True, help="data/manifests/images.csv")
    ap.add_argument("--patches", type=Path, required=True, help="data/manifests/patches.csv")
    ap.add_argument("--out_json", type=Path, required=False, default=None, help="splits.json path")
    ap.add_argument("--out_patches", type=Path, required=False, default=None, help="patch manifest with split column")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    images_df = pd.read_csv(args.images)
    patches_df = pd.read_csv(args.patches)

    splits = make_image_splits(images_df, seed=args.seed, ratios=(args.train, args.val, args.test))
    out_json = args.out_json or (Path(args.patches).parent / "splits.json")
    out_patches = args.out_patches or (Path(args.patches).parent / "patches_splits.csv")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_patches.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(splits, f, indent=2)

    tagged = tag_patches_with_split(patches_df, splits)
    tagged.to_csv(out_patches, index=False)

    # Console summary
    counts = tagged["split"].value_counts().to_dict()
    parent_counts = {k: len(v) for k, v in splits.items()}

    print(f"[info] Wrote splits: {out_json}")
    print(f"[info] Wrote patched manifest with splits: {out_patches} ({len(tagged)} rows)")
    print(f"[info] Split counts (tiles): {counts}")
    print(f"[info] Parent images per split: {parent_counts}")

if __name__ == "__main__":
    main()
