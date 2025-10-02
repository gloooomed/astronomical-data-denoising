from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="QC for patches and splits.")
    ap.add_argument("--patch_manifest", type=Path, required=True, help="data/manifests/patches_splits.csv")
    ap.add_argument("--splits_json", type=Path, required=True, help="data/manifests/splits.json")
    args = ap.parse_args()

    df = pd.read_csv(args.patch_manifest)
    with open(args.splits_json) as f:
        splits = json.load(f)

    # 1) Each parent must map to exactly one split
    parent_split = df.groupby("parent_id")["split"].nunique()
    bad = parent_split[parent_split != 1]
    if len(bad):
        print("[warn] Some parents appear in multiple splits:", bad.index.tolist())
    else:
        print("[ok] Each parent appears in exactly one split.")

    # 2) Tiles per parent distribution (by split)
    tpp = df.groupby(["split", "parent_id"]).size().reset_index(name="tiles")
    summary = tpp.groupby("split")["tiles"].describe()
    print("\n[info] Tiles per parent (by split):")
    print(summary.to_string())

    # 3) Channel and size consistency
    print("\n[info] Channels distribution:", df["c"].value_counts().to_dict())
    print("[info] Tile size check (unique H,W pairs):")
    print(df[["h","w"]].drop_duplicates().head())

    # 4) Count check
    print("\n[info] Total tiles:", len(df))
    print("[info] Split counts (tiles):", df["split"].value_counts().to_dict())

if __name__ == "__main__":
    main()
