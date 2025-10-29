# src/eval/metrics_and_plots.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from sklearn.metrics import precision_recall_curve, confusion_matrix
# photutils/astropy import with fallbacks
try:
    # new-ish layout
    from photutils.detection import DAOStarFinder
except Exception:
    try:
        # older layout
        from photutils import DAOStarFinder
    except Exception as e:
        raise ImportError(
            "DAOStarFinder not found in photutils. Please (re)install photutils and astropy.\n"
            "Install with: pip install photutils astropy"
        ) from e

try:
    from astropy.stats import sigma_clipped_stats
except Exception as e:
    raise ImportError("astropy not found. Install with: pip install astropy") from e

from astropy.table import Table

import torch
from PIL import Image
from models.unet_blindspot import UNetBlindspot

sns.set(style="whitegrid")

# ---------------------------
# Utilities: metrics & IO
# ---------------------------
def load_ckpt_model(ckpt_path: Path, device: str = "cpu", base: int = 48) -> torch.nn.Module:
    model = UNetBlindspot(in_ch=3, base=base).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model

def read_patch(path: str) -> np.ndarray:
    arr = np.load(path) if path.endswith(".npy") else np.asarray(Image.open(path)).astype("float32")/255.0
    if arr.dtype == np.uint8:
        arr = arr.astype("float32")/255.0
    return arr.astype("float32")  # H,W,C, in [0,1]

def compute_pixel_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Compute basic pixel-wise metrics and a robust SSIM that handles small images.
    target, pred : H,W,C in [0,1]
    returns dict with MSE, MAE, PSNR, SSIM (SSIM may be NaN if image too small)
    """
    # Ensure shapes match
    assert target.shape == pred.shape, f"target/pred shape mismatch: {target.shape} vs {pred.shape}"

    mse = float(np.mean((target - pred) ** 2))
    mae = float(np.mean(np.abs(target - pred)))
    # PSNR from skimage expects data_range
    try:
        psnr = float(sk_psnr(target, pred, data_range=1.0))
    except Exception:
        psnr = float(np.nan)

    # Robust SSIM: choose a valid odd win_size <= min(H,W)
    H, W = target.shape[0], target.shape[1]
    min_side = min(H, W)
    ssim_val = float(np.nan)
    try:
        # default desired window
        desired_win = 7
        if min_side < 3:
            # too small for meaningful SSIM
            ssim_val = float(np.nan)
        else:
            # pick largest odd win <= min_side and <= desired_win
            win = min(desired_win, min_side)
            # make odd
            if win % 2 == 0:
                win -= 1
            if win < 3:
                # fallback: can't compute ssim reliably
                ssim_val = float(np.nan)
            else:
                # modern skimage uses channel_axis
                try:
                    ssim_val = float(sk_ssim(target, pred, data_range=1.0, channel_axis=2, win_size=win))
                except TypeError:
                    # older skimage versions expect multichannel arg
                    ssim_val = float(sk_ssim(target, pred, data_range=1.0, multichannel=True, win_size=win))
    except Exception:
        ssim_val = float(np.nan)

    return dict(MSE=mse, MAE=mae, PSNR=psnr, SSIM=ssim_val)


# ---------------------------
# Astronomy-aware helpers
# ---------------------------
def detect_sources(image: np.ndarray, fwhm: float = 3.0, threshold_sigma: float = 3.0):
    """
    Robust source detection returning an astropy Table with columns ['x','y','flux'].
    Uses DAOStarFinder and adapts to different photutils/astropy column names.
    """
    # Convert to grayscale luminance if needed
    if image.ndim == 3:
        gray = 0.299*image[...,0] + 0.587*image[...,1] + 0.114*image[...,2]
    else:
        gray = image

    # Compute background stats
    try:
        mean, median, std = sigma_clipped_stats(gray, sigma=3.0)
    except Exception:
        # fallback: simple stats
        mean, median, std = float(np.mean(gray)), float(np.median(gray)), float(np.std(gray))

    # Run DAOStarFinder (may return None)
    try:
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma*std)
        sources = daofind(gray - median)
    except Exception as e:
        # If DAOStarFinder fails for any reason, return empty table
        from astropy.table import Table
        return Table(names=("x","y","flux"), rows=[])

    if sources is None or len(sources) == 0:
        from astropy.table import Table
        return Table(names=("x","y","flux"), rows=[])

    # Possible column name patterns for x,y,flux in various photutils versions:
    possible_x_cols = ['x', 'xcentroid', 'x_peak', 'x_0', 'xcenter', 'xcen']
    possible_y_cols = ['y', 'ycentroid', 'y_peak', 'y_0', 'ycenter', 'ycen']
    possible_flux_cols = ['flux', 'flux_0', 'flux_peak', 'aperture_sum', 'flux_fit']

    # find actual columns
    cols = sources.colnames
    def find_col(candidates):
        for c in candidates:
            if c in cols:
                return c
        # try partial matches
        for c in cols:
            lc = c.lower()
            for cand in candidates:
                if cand in lc:
                    return c
        return None

    xcol = find_col(possible_x_cols)
    ycol = find_col(possible_y_cols)
    fcol = find_col(possible_flux_cols)

    # If any not found, try other heuristics
    if xcol is None or ycol is None:
        # try columns that look numeric and have names with 'x' or 'y'
        for c in cols:
            if xcol is None and ('x' in c.lower() or 'colx' in c.lower()):
                xcol = c
            if ycol is None and ('y' in c.lower() or 'coly' in c.lower()):
                ycol = c

    # If flux not found, try any numeric column that's not x/y and pick the largest-variance one
    if fcol is None:
        numeric_cols = [c for c in cols if np.issubdtype(sources[c].dtype, np.number) and c not in (xcol, ycol)]
        if numeric_cols:
            # pick column with highest variance
            varvals = [(c, np.var(sources[c])) for c in numeric_cols]
            varvals.sort(key=lambda t: t[1], reverse=True)
            fcol = varvals[0][0]

    # Build a cleaned table with x,y,flux; if something missing, fill with zeros
    from astropy.table import Table, Column
    out = Table()
    n = len(sources)
    # x
    if xcol is not None:
        out['x'] = Column(sources[xcol].astype(float))
    else:
        out['x'] = Column(np.zeros(n, dtype=float))
    # y
    if ycol is not None:
        out['y'] = Column(sources[ycol].astype(float))
    else:
        out['y'] = Column(np.zeros(n, dtype=float))
    # flux
    if fcol is not None:
        out['flux'] = Column(sources[fcol].astype(float))
    else:
        # fallback: approximate flux by taking the value at integer coords from image
        fluxs = []
        for xi, yi in zip(out['x'], out['y']):
            xi_i = int(round(xi)) if np.isfinite(xi) else 0
            yi_i = int(round(yi)) if np.isfinite(yi) else 0
            if 0 <= yi_i < gray.shape[0] and 0 <= xi_i < gray.shape[1]:
                fluxs.append(float(gray[yi_i, xi_i]))
            else:
                fluxs.append(0.0)
        out['flux'] = Column(np.array(fluxs, dtype=float))

    return out


def match_sources(ref: Table, cand: Table, max_sep: float = 3.0) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Match cand -> ref with nearest neighbor within max_sep (pixels).
    Returns:
      matches: list of (ref_idx, cand_idx)
      unmatched_ref: list of ref indices without match (false negatives)
      unmatched_cand: list of cand indices without match (false positives)
    """
    if len(ref) == 0 and len(cand) == 0:
        return [], [], []
    if len(ref) == 0:
        return [], [], list(range(len(cand)))
    if len(cand) == 0:
        return [], list(range(len(ref))), []

    ref_coords = np.vstack((ref['x'], ref['y'])).T
    cand_coords = np.vstack((cand['x'], cand['y'])).T

    # brute-force NN (small lists)
    matches = []
    used_cand = set()
    for i, rc in enumerate(ref_coords):
        dists = np.linalg.norm(cand_coords - rc[None, :], axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= max_sep:
            matches.append((i, j))
            used_cand.add(j)
    unmatched_ref = [i for i in range(len(ref)) if i not in [m[0] for m in matches]]
    unmatched_cand = [j for j in range(len(cand)) if j not in used_cand]
    return matches, unmatched_ref, unmatched_cand

# ---------------------------
# Plotting helpers
# ---------------------------
def plot_loss_curve(train_csv: Path, out: Path):
    """
    train_csv: CSV with columns ['epoch','train_loss','val_loss'] or similar.
    """
    df = pd.read_csv(train_csv)
    plt.figure(figsize=(7,4))
    if 'train_loss' in df.columns:
        plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title('Loss vs Epoch')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_metric_hist(metric_vals: np.ndarray, title: str, out: Path, bins: int = 40):
    plt.figure(figsize=(6,4))
    plt.hist(metric_vals, bins=bins, density=False)
    plt.xlabel(title); plt.ylabel('Count'); plt.title(f'{title} distribution')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_psnr_ssim_scatter(psnrs: np.ndarray, ssims: np.ndarray, out: Path):
    plt.figure(figsize=(6,6))
    plt.scatter(psnrs, ssims, s=10, alpha=0.6)
    plt.xlabel('PSNR (dB)'); plt.ylabel('SSIM'); plt.title('PSNR vs SSIM')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

def plot_psnr_sorted(psnrs: np.ndarray, out: Path):
    s = np.sort(psnrs)[::-1]
    plt.figure(figsize=(7,4))
    plt.plot(s, marker='.', linestyle='none')
    plt.xlabel('image rank'); plt.ylabel('PSNR (dB)'); plt.title('PSNR per image (sorted)')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_flux_scatter(true_flux: np.ndarray, pred_flux: np.ndarray, out: Path):
    plt.figure(figsize=(6,6))
    plt.scatter(true_flux, pred_flux, s=6, alpha=0.4)
    mx = max(true_flux.max(), pred_flux.max())
    plt.plot([0,mx],[0,mx], 'r--', label='y=x')
    plt.xlabel('True flux'); plt.ylabel('Denoised flux'); plt.title('True vs Denoised flux (matched sources)')
    plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_confusion_matrix(tp: int, fp: int, fn: int, out: Path):
    cm = np.array([[tp, fp],[fn, 0]])  # [TP FP; FN ?] (note: TN is undefined here)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Detected','Not Detected'], yticklabels=['GroundTruth','GroundTruthMissing'])
    plt.title('Detection counts (aggregated)')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

# ---------------------------
# Main evaluation routine
# ---------------------------
def evaluate_and_plot(args):
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    model = load_ckpt_model(Path(args.ckpt), device=device, base=args.base)

    patches_df = pd.read_csv(args.patch_csv)
    test_df = patches_df[patches_df['split'] == 'test'].reset_index(drop=True)

    per_patch_results = []
    all_true_flux = []
    all_pred_flux = []
    total_tp = total_fp = total_fn = 0

    # sample n_images full parents (if specified)
    parents = test_df['parent_id'].unique().tolist()
    if args.max_parents:
        parents = parents[:args.max_parents]

    # For each parent, collect tiles -> reconstruct full image arrays for GT and denoised
    for pid in tqdm(parents, desc="Parents"):
        sub = test_df[test_df['parent_id'] == pid].copy()
        # Reconstruct canvas shape
        h = int(sub['h'].iloc[0]); w = int(sub['w'].iloc[0])
        y_max = int(sub['y'].max()); x_max = int(sub['x'].max())
        H = y_max + h; W = x_max + w

        # prepare empty arrays for GT (from saved .npy) and denoised canvas accumulators
        gt_canvas = np.zeros((H, W, 3), dtype=np.float32)
        count_canvas = np.zeros((H, W, 1), dtype=np.float32)
        pred_canvas = np.zeros_like(gt_canvas)

        tiles_coords = []
        # first pass: read patches and build GT canvas (averaging overlaps)
        for row in sub.itertuples():
            arr = read_patch(row.path)  # H,W,C
            y, x = int(row.y), int(row.x)
            gt_canvas[y:y+arr.shape[0], x:x+arr.shape[1], :] += arr
            count_canvas[y:y+arr.shape[0], x:x+arr.shape[1], :] += 1.0
            tiles_coords.append((row.path, (y,x)))

        count_canvas = np.clip(count_canvas, 1.0, None)
        gt_canvas = gt_canvas / count_canvas  # averaged GT canvas

        # second pass: run model on each tile, place into pred_canvas (simple average)
        pred_acc = np.zeros_like(pred_canvas)
        pred_count = np.zeros_like(count_canvas)
        for path, (y,x) in tiles_coords:
            arr = read_patch(path)
            inp = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).to(device)  # 1,C,H,W
            with torch.no_grad():
                out = model(inp).clamp(0,1).squeeze(0).cpu().numpy().transpose(1,2,0)
            pred_acc[y:y+out.shape[0], x:x+out.shape[1], :] += out
            pred_count[y:y+out.shape[0], x:x+out.shape[1], :] += 1.0

        pred_count = np.clip(pred_count, 1.0, None)
        pred_canvas = pred_acc / pred_count

        # Compute pixel metrics on the full canvas
        pixm = compute_pixel_metrics(gt_canvas, pred_canvas)
        pixm['parent_id'] = pid
        per_patch_results.append(pixm)

        # Astronomy: detect in GT and pred canvases
        gt_sources = detect_sources(gt_canvas, fwhm=args.fwhm, threshold_sigma=args.threshold_sigma)
        pred_sources = detect_sources(pred_canvas, fwhm=args.fwhm, threshold_sigma=args.threshold_sigma)

        matches, unmatched_ref, unmatched_cand = match_sources(gt_sources, pred_sources, max_sep=args.max_sep)
        total_tp += len(matches)
        total_fn += len(unmatched_ref)
        total_fp += len(unmatched_cand)

        # collect matched flux pairs
        for (i_ref, i_cand) in matches:
            all_true_flux.append(float(gt_sources['flux'][i_ref]))
            all_pred_flux.append(float(pred_sources['flux'][i_cand]))

    # Save per-parent metrics CSV
    metrics_df = pd.DataFrame(per_patch_results)
    metrics_df.to_csv(Path(args.out_dir)/"per_parent_metrics.csv", index=False)

    # Aggregate pixel metrics arrays for plotting
    psnrs = metrics_df['PSNR'].values
    ssims = metrics_df['SSIM'].values

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Loss curve plot (if provided)
    if args.train_history is not None:
        plot_loss_curve(Path(args.train_history), outdir/"loss_vs_epoch.png")

    # 2) PSNR histogram
    plot_metric_hist(psnrs, "PSNR (dB)", outdir/"psnr_hist.png")
    plot_psnr_sorted(psnrs, outdir/"psnr_sorted.png")
    plot_metric_hist(ssims, "SSIM", outdir/"ssim_hist.png")
    plot_psnr_ssim_scatter(psnrs, ssims, outdir/"psnr_vs_ssim.png")

    # 3) Flux scatter & residual
    if len(all_true_flux):
        plot_flux_scatter(np.array(all_true_flux), np.array(all_pred_flux), outdir/"flux_scatter.png")
        # residual histogram
        residuals = np.array(all_pred_flux) - np.array(all_true_flux)
        plot_metric_hist(residuals, "Flux residual (pred-true)", outdir/"flux_residual_hist.png")

    # 4) Detection confusion & PR curve
    tp, fp, fn = total_tp, total_fp, total_fn
    plot_confusion_matrix(tp, fp, fn, outdir/"detection_confusion.png")
    # precision / recall
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec = tp / (tp + fn) if (tp+fn)>0 else 0.0
    with open(outdir/"detection_summary.txt","w") as f:
        f.write(f"TP {tp}\\nFP {fp}\\nFN {fn}\\nPrecision {prec:.4f}\\nRecall {rec:.4f}\\n")

    # 5) Save aggregated metrics
    metrics_df.to_csv(outdir/"per_parent_metrics.csv", index=False)
    print(f"[info] Saved plots and metrics to {outdir}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="reports/eval")
    ap.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--max_parents", type=int, default=20, help="Max parents to evaluate (for speed)")
    # detection params
    ap.add_argument("--fwhm", type=float, default=3.0)
    ap.add_argument("--threshold_sigma", type=float, default=3.0)
    ap.add_argument("--max_sep", type=float, default=3.0)
    ap.add_argument("--train_history", type=str, default=None, help="Optional CSV of train/val loss per epoch")
    args = ap.parse_args()
    evaluate_and_plot(args)
