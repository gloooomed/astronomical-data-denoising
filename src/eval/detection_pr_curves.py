# src/eval/detection_pr_curves.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import torch
from PIL import Image

# Reuse robust helpers (copy/paste small required helpers to avoid import issues)
def read_patch(path: str) -> np.ndarray:
    p = path
    if not Path(p).exists():
        alt = Path.cwd() / Path(p)
        if alt.exists():
            p = str(alt)
        else:
            raise FileNotFoundError(f"Cannot find {path} or {alt}")
    arr = np.load(p) if p.endswith(".npy") else np.asarray(Image.open(p)).astype("float32") / 255.0
    if arr.dtype == np.uint8:
        arr = arr.astype("float32") / 255.0
    return arr.astype("float32")

# robust SSIM wrapper (small)
def safe_ssim(a, b):
    try:
        H, W = a.shape[:2]
        win = 7
        if min(H, W) < 7:
            win = min(7, min(H, W))
            if win % 2 == 0:
                win -= 1
        if win < 3:
            return float("nan")
        try:
            return float(sk_ssim(a, b, data_range=1.0, channel_axis=2, win_size=win))
        except TypeError:
            return float(sk_ssim(a, b, data_range=1.0, multichannel=True, win_size=win))
    except Exception:
        return float("nan")

# Simple detector fallback using photutils if available
try:
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table, Column
except Exception:
    DAOStarFinder = None
    sigma_clipped_stats = None
    Table = None

def detect_sources_gray(image: np.ndarray, fwhm: float = 3.0, threshold_sigma: float = 3.0):
    # grayscale luminance
    if image.ndim == 3:
        gray = 0.299*image[...,0] + 0.587*image[...,1] + 0.114*image[...,2]
    else:
        gray = image
    if DAOStarFinder is None or sigma_clipped_stats is None:
        # fallback: detect local maxima using a simple method (threshold + local max)
        med = np.median(gray)
        std = np.std(gray - med)
        th = med + threshold_sigma * std
        # local maxima
        from scipy import ndimage
        im_max = ndimage.maximum_filter(gray, size=3)
        peaks = (gray == im_max) & (gray > th)
        ys, xs = np.nonzero(peaks)
        flux = gray[ys, xs]
        if len(xs) == 0:
            return []  # empty list
        return list(zip(xs.tolist(), ys.tolist(), flux.tolist()))
    else:
        mean, median, std = sigma_clipped_stats(gray, sigma=3.0)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma*std)
        sources = daofind(gray - median)
        if sources is None or len(sources) == 0:
            return []
        # attempt to find x,y,flux columns robustly
        cols = sources.colnames
        def pick(colopts):
            for c in colopts:
                if c in cols:
                    return c
            for c in cols:
                lc = c.lower()
                for cand in colopts:
                    if cand in lc:
                        return c
            return None
        xcol = pick(['x','xcentroid','x_peak','x_0'])
        ycol = pick(['y','ycentroid','y_peak','y_0'])
        fcol = pick(['flux','flux_0','flux_peak','aperture_sum'])
        res = []
        for row in sources:
            x = float(row[xcol]) if xcol in cols else float(row[0])
            y = float(row[ycol]) if ycol in cols else float(row[1])
            flux = float(row[fcol]) if (fcol and fcol in cols) else 0.0
            res.append((x,y,flux))
        return res

def match_sources_list(ref_list, cand_list, max_sep=3.0):
    # inputs: lists of (x,y,flux)
    if len(ref_list) == 0 and len(cand_list) == 0:
        return [], list(), list()
    ref = np.array([[r[0], r[1]] for r in ref_list]) if len(ref_list) else np.zeros((0,2))
    cand = np.array([[c[0], c[1]] for c in cand_list]) if len(cand_list) else np.zeros((0,2))
    matches = []
    used_cand = set()
    for i, rc in enumerate(ref):
        if cand.shape[0]==0:
            continue
        dists = np.linalg.norm(cand - rc[None,:], axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= max_sep:
            matches.append((i, j))
            used_cand.add(j)
    unmatched_ref = [i for i in range(len(ref_list)) if i not in [m[0] for m in matches]]
    unmatched_cand = [j for j in range(len(cand_list)) if j not in used_cand]
    return matches, unmatched_ref, unmatched_cand

# ---------------------------
# Main script
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="reports/eval_detection")
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--max_parents", type=int, default=20)
    ap.add_argument("--fwhm", type=float, default=3.0)
    ap.add_argument("--max_sep", type=float, default=3.0)
    ap.add_argument("--thresholds", type=str, default="1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu"

    # load manifest
    df = pd.read_csv(args.patch_csv)
    test_df = df[df['split']=='test'].reset_index(drop=True)
    parents = test_df['parent_id'].unique().tolist()
    if args.max_parents and args.max_parents>0:
        parents = parents[:args.max_parents]

    # load model
    ckpt = torch.load(args.ckpt, map_location=device)
    from models.unet_blindspot import UNetBlindspot
    model = UNetBlindspot(in_ch=3, base=48).to(device).eval()
    model.load_state_dict(ckpt['model'])

    # thresholds to sweep
    thresholds = [float(x) for x in args.thresholds.split(',')]

    # precompute full canvases to speed repeated detection runs
    canvases = []  # list of tuples (pid, gt_canvas, pred_canvas)
    print("Reconstructing canvases (this may take a while)...")
    for pid in tqdm(parents, desc="Parents"):
        sub = test_df[test_df['parent_id']==pid]
        h = int(sub['h'].iloc[0]); w = int(sub['w'].iloc[0])
        y_max = int(sub['y'].max()); x_max = int(sub['x'].max())
        H = y_max + h; W = x_max + w

        gt_canvas = np.zeros((H,W,3), dtype=np.float32)
        count_canvas = np.zeros((H,W,1), dtype=np.float32)
        tiles = []
        coords = []

        for row in sub.itertuples():
            arr = read_patch(row.path)
            y = int(row.y); x = int(row.x)
            gt_canvas[y:y+arr.shape[0], x:x+arr.shape[1], :] += arr
            count_canvas[y:y+arr.shape[0], x:x+arr.shape[1], :] += 1.0
            tiles.append((row.path, (y,x)))

        count_canvas = np.clip(count_canvas, 1.0, None)
        gt_canvas = gt_canvas / count_canvas

        # predict per-tile and aggregate (simple average)
        pred_acc = np.zeros_like(gt_canvas)
        pred_count = np.zeros_like(count_canvas)
        for path, (y,x) in tiles:
            arr = read_patch(path)
            inp = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp).clamp(0,1).squeeze(0).cpu().numpy().transpose(1,2,0)
            pred_acc[y:y+out.shape[0], x:x+out.shape[1], :] += out
            pred_count[y:y+out.shape[0], x:x+out.shape[1], :] += 1.0
        pred_count = np.clip(pred_count, 1.0, None)
        pred_canvas = pred_acc / pred_count

        canvases.append((pid, gt_canvas, pred_canvas))

    # compute PSNR/SSIM per parent
    psnrs = []
    ssims = []
    for pid, gt, pred in canvases:
        psnrs.append(float(sk_psnr(gt, pred, data_range=1.0)))
        ssims.append(safe_ssim(gt, pred))

    # plot PSNR/SSIM
    sns.set(style="whitegrid")
    plt.figure(figsize=(6,4)); plt.hist(psnrs, bins=30); plt.xlabel("PSNR (dB)"); plt.title("PSNR histogram"); plt.tight_layout(); plt.savefig(outdir/"psnr_hist.png"); plt.close()
    plt.figure(figsize=(6,4)); plt.hist([s for s in ssims if not np.isnan(s)], bins=30); plt.xlabel("SSIM"); plt.title("SSIM histogram"); plt.tight_layout(); plt.savefig(outdir/"ssim_hist.png"); plt.close()
    plt.figure(figsize=(6,6)); plt.scatter(psnrs, ssims, s=8, alpha=0.6); plt.xlabel("PSNR"); plt.ylabel("SSIM"); plt.title("PSNR vs SSIM"); plt.tight_layout(); plt.savefig(outdir/"psnr_vs_ssim.png"); plt.close()

    # sweep thresholds and accumulate TP/FP/FN
    results = []
    for thresh in thresholds:
        tp = fp = fn = 0
        for pid, gt, pred in canvases:
            gt_src = detect_sources_gray(gt, fwhm=args.fwhm, threshold_sigma=thresh)
            pred_src = detect_sources_gray(pred, fwhm=args.fwhm, threshold_sigma=thresh)
            matches, unmatched_ref, unmatched_cand = match_sources_list(gt_src, pred_src, max_sep=args.max_sep)
            tp += len(matches)
            fn += len(unmatched_ref)
            fp += len(unmatched_cand)
        precision = tp / (tp + fp) if (tp+fp)>0 else 0.0
        recall = tp / (tp + fn) if (tp+fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        results.append((thresh, tp, fp, fn, precision, recall, f1))

    # convert to DataFrame
    rdf = pd.DataFrame(results, columns=["threshold","TP","FP","FN","precision","recall","f1"])
    rdf.to_csv(outdir/"detection_threshold_sweep.csv", index=False)

    # plots:
    # PR curve
    plt.figure(figsize=(6,6))
    plt.plot(rdf['recall'], rdf['precision'], marker='o')
    for i,r in rdf.iterrows():
        plt.text(r['recall'], r['precision'], f"{r['threshold']:.1f}", fontsize=8)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (threshold labels)"); plt.grid(True); plt.tight_layout(); plt.savefig(outdir/"pr_curve.png"); plt.close()

    # precision/recall/f1 vs threshold
    plt.figure(figsize=(7,4))
    plt.plot(rdf['threshold'], rdf['precision'], label='precision', marker='o')
    plt.plot(rdf['threshold'], rdf['recall'], label='recall', marker='o')
    plt.plot(rdf['threshold'], rdf['f1'], label='f1', marker='o')
    plt.xlabel("detection threshold (sigma)"); plt.ylabel("Score"); plt.legend(); plt.title("Precision / Recall / F1 vs threshold"); plt.tight_layout(); plt.savefig(outdir/"prf_vs_threshold.png"); plt.close()

    # detection counts vs threshold
    plt.figure(figsize=(7,4))
    plt.plot(rdf['threshold'], rdf['TP'], label='TP', marker='o')
    plt.plot(rdf['threshold'], rdf['FP'], label='FP', marker='o')
    plt.plot(rdf['threshold'], rdf['FN'], label='FN', marker='o')
    plt.xlabel("detection threshold (sigma)"); plt.ylabel("counts"); plt.legend(); plt.title("Detection counts vs threshold"); plt.tight_layout(); plt.savefig(outdir/"detection_counts_vs_threshold.png"); plt.close()

    # summary
    best_idx = rdf['f1'].idxmax()
    best_row = rdf.loc[best_idx]
    with open(outdir/"summary_metrics.txt","w") as f:
        f.write(f"best threshold (by F1): {best_row['threshold']}\n")
        f.write(f"Precision: {best_row['precision']:.4f}\n")
        f.write(f"Recall: {best_row['recall']:.4f}\n")
        f.write(f"F1: {best_row['f1']:.4f}\n")
        f.write("\nfull table saved to detection_threshold_sweep.csv\n")

    print(f"[info] wrote detection plots & CSV to {outdir}")

if __name__ == "__main__":
    import argparse, seaborn as sns
    sns.set(style="whitegrid")
    main()
