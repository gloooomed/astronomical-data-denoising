# src/dataio/manifest.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# --------- Utilities

IMG_EXTS = {".jpg", ".jpeg", ".png"}  # Extend later if needed
TELESCOPE_PATTERNS = [
    (re.compile(r"\bhst\b", re.IGNORECASE), "HST"),
    (re.compile(r"\bjwst\b", re.IGNORECASE), "JWST"),
]


def sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-1 of a file; used for dedupe and stable IDs."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def infer_telescope_and_band(path: Path) -> Tuple[str, str]:
    """
    Infer telescope (and optionally band) from the path/filename.
    Keep this conservative; return '' when unsure.
    """
    s = str(path).replace(os.sep, "/")  # normalize
    telescope = ""
    for pat, name in TELESCOPE_PATTERNS:
        if pat.search(s):
            telescope = name
            break

    # Band inference intentionally left blank for now to avoid errors.
    band = ""
    return telescope, band


def channels_from_mode(mode: str) -> int:
    # Pillow modes: "L" (1), "RGB" (3), "RGBA" (4), "I;16" etc.
    if mode == "L":
        return 1
    if mode == "RGB":
        return 3
    if mode == "RGBA":
        return 4
    # Fallback: common cases
    if mode.startswith("I;") or mode in {"I", "F"}:
        return 1
    return 1  # default safe fallback


@dataclass
class ImageRow:
    id: str
    sha1: str
    path: str
    filename: str
    fmt: str
    mode: str
    width: int
    height: int
    channels: int
    telescope: str
    band: str
    bytes: int
    mtime: float


def is_image_path(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def discover_files(root: Path) -> List[Path]:
    """Recursively find candidate image files under root."""
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")
    return [p for p in root.rglob("*") if p.is_file() and is_image_path(p)]


def read_pillow_meta(path: Path) -> Tuple[str, str, int, int]:
    """
    Read lightweight metadata via Pillow without decoding full pixel data.
    Returns (fmt, mode, width, height).
    """
    with Image.open(path) as im:
        fmt = im.format or ""
        mode = im.mode or ""
        width, height = im.size
    return fmt, mode, width, height


def build_manifest(root: Path) -> pd.DataFrame:
    rows: List[ImageRow] = []
    files = discover_files(root)

    for path in tqdm(files, desc="Scanning images"):
        try:
            sha1 = sha1_file(path)
            short_id = sha1[:12]
            fmt, mode, width, height = read_pillow_meta(path)
            telescope, band = infer_telescope_and_band(path)
            stat = path.stat()

            row = ImageRow(
                id=short_id,
                sha1=sha1,
                path=str(path.resolve()),
                filename=path.name,
                fmt=fmt,
                mode=mode,
                width=int(width),
                height=int(height),
                channels=int(channels_from_mode(mode)),
                telescope=telescope,
                band=band,
                bytes=int(stat.st_size),
                mtime=float(stat.st_mtime),
            )
            rows.append(row)

        except UnidentifiedImageError:
            # Corrupted or unsupported image file; skip but warn.
            print(f"[warn] Unidentified image: {path}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] Failed {path}: {e}", file=sys.stderr)

    df = pd.DataFrame([asdict(r) for r in rows])
    return df


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    out["count"] = int(len(df))
    out["formats"] = df["fmt"].value_counts().to_dict()
    out["modes"] = df["mode"].value_counts().to_dict()
    out["channels"] = df["channels"].value_counts().to_dict()
    out["min_size"] = {
        "width": int(df["width"].min()),
        "height": int(df["height"].min()),
    } if len(df) else {"width": 0, "height": 0}
    out["max_size"] = {
        "width": int(df["width"].max()),
        "height": int(df["height"].max()),
    } if len(df) else {"width": 0, "height": 0}
    out["telescope_counts"] = df["telescope"].replace("", "UNK").value_counts().to_dict()

    # Duplicate detection by sha1
    dupes = df["sha1"].duplicated(keep=False)
    out["duplicates"] = int(dupes.sum())
    return out


def save_outputs(df: pd.DataFrame, out_csv: Path, out_parquet: Optional[Path] = None) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if out_parquet is not None:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out_parquet, index=False)
        except Exception as e:
            print(f"[warn] Parquet save failed ({e}); CSV is still written.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Build raw image manifest (JPG/PNG).")
    parser.add_argument("--root", type=Path, required=True, help="Folder containing raw images")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--parquet", type=Path, default=None, help="Optional Parquet output path")
    parser.add_argument("--print-summary", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args()

    df = build_manifest(args.root)

    # Sort by path for stable diffs
    if len(df):
        df = df.sort_values("path").reset_index(drop=True)

    save_outputs(df, args.out, args.parquet)
    summary = summarize(df)

    print(f"[info] Wrote manifest: {args.out}  ({len(df)} rows)")
    if args.parquet:
        print(f"[info] Wrote Parquet:  {args.parquet}")

    if args.print_summary:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
