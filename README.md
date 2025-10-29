# Astronomical Data Denoising

Self-supervised denoising of astronomical images using Noise2Void with blind-spot U-Net training. This project provides a complete pipeline for denoising astronomical observations without requiring clean reference images.

## Overview

This project implements a self-supervised deep learning approach for denoising astronomical images using the Noise2Void method with a blind-spot U-Net architecture. The system is designed specifically for astronomical data with features like:

- **Self-Supervised Learning**: No clean reference images required
- **Blind-Spot Training**: Random 5×5 pixel regions masked during training
- **Astronomy-Aware**: Preserves flux, detects sources, computes PSNR/SSIM metrics
- **Production-Ready**: Early stopping, AMP support, PyTorch 2.x compatibility
- **Complete Pipeline**: From raw images to denoised outputs with evaluation

## Features

### Core Capabilities
- **Advanced Training**: Early stopping, mixed precision (AMP), gradient scaling, and PyTorch 2.x compatibility
- **Data Pipeline**: Automated manifest creation, percentile-based normalization, overlapping tile extraction
- **Smart Stitching**: Cosine windowing for seamless reconstruction from overlapping patches
- **Astronomy Tools**: Star detection (DAOStarFinder), flux preservation analysis, PSNR/SSIM evaluation
- **Interactive Analysis**: GUI-based image selection with detailed JSON reports and visualizations

### Inference & Evaluation
- Single-patch preview denoising for quick validation
- Full-image reconstruction with overlap handling
- Comprehensive evaluation suite with precision-recall curves
- Detection threshold sweeping for optimal parameter selection
- Automated star counting and flux comparison

## Project Structure

```
astronomical-data-denoising/
├── src/
│   ├── train_n2v.py              # Main training script with early stopping
│   ├── predict_and_analyze.py    # Interactive GUI prediction + analysis
│   ├── models/
│   │   └── unet_blindspot.py     # U-Net with skip connections (base=48)
│   ├── losses/
│   │   └── masked_loss.py        # Blind-spot masking and masked L2 loss
│   ├── dataio/
│   │   ├── manifest.py           # Build image catalogs with SHA-1 IDs
│   │   ├── tiler.py              # Extract 512×512 patches with overlap
│   │   ├── split.py              # Parent-level train/val/test splits
│   │   ├── qc.py                 # Quality control validation
│   │   └── loaders.py            # PyTorch DataLoader with augmentations
│   ├── utils/
│   │   └── stitch.py             # Cosine-weighted tile stitching
│   ├── tools/
│   │   ├── preview_denoise.py    # Preview denoising on test patches
│   │   └── denoise_full.py       # Reconstruct full denoised images
│   └── eval/
│       ├── metrics_and_plots.py  # Comprehensive evaluation suite
│       └── detection_pr_curves.py # Precision-recall analysis
├── scripts/
│   ├── sample_manifest.py        # Sample subset of manifest
│   └── test_loader.py            # Test data loader functionality
├── data/
│   ├── raw/                      # Original astronomical images
│   ├── patches_50/               # Extracted 512×512 patches (.npy)
│   └── manifests/                # CSV manifests and split definitions
├── checkpoints/
│   └── n2v_unet/                 # Saved model checkpoints
├── reports/
│   ├── preview/                  # Preview denoising outputs
│   ├── predictions/              # Interactive prediction results
│   ├── eval_quick/               # Quick evaluation metrics
│   └── eval_detection/           # Detection threshold sweeps
├── requirements.txt              # Python dependencies
└── README.md
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/gloooomed/astronomical-data-denoising.git
cd astronomical-data-denoising

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=8.0.0
tqdm>=4.60.0
```

Optional for evaluation:
```bash
pip install scikit-image scikit-learn matplotlib seaborn photutils astropy scipy
```

## Usage

### 1. Data Preparation

#### Build Image Manifest
Scan a directory of raw astronomical images and create a catalog with metadata:

```bash
python src/dataio/manifest.py \
    --root data/raw \
    --out data/manifests/images.csv \
    --print-summary
```

Output: CSV with columns: `id`, `sha1`, `path`, `filename`, `fmt`, `mode`, `width`, `height`, `channels`, `telescope`, `band`, `bytes`, `mtime`

#### Extract Patches
Tile images into 512×512 patches with 50% overlap and percentile normalization:

```bash
python src/dataio/tiler.py \
    --manifest data/manifests/images.csv \
    --outdir data/patches_50 \
    --patch_manifest data/manifests/patches_50.csv \
    --tile 512 \
    --stride 256 \
    --p_low 1.0 \
    --p_high 99.5
```

Output: `.npy` files (float16) and a patch manifest with normalization metadata.

#### Create Splits
Generate train/val/test splits at the parent image level:

```bash
python src/dataio/split.py \
    --images data/manifests/images.csv \
    --patches data/manifests/patches_50.csv \
    --out_json data/manifests/splits_50.json \
    --out_patches data/manifests/patches_50_splits.csv \
    --train 0.70 --val 0.15 --test 0.15 \
    --seed 42
```

#### Quality Control
Verify splits and patch consistency:

```bash
python src/dataio/qc.py \
    --patch_manifest data/manifests/patches_50_splits.csv \
    --splits_json data/manifests/splits_50.json
```

### 2. Training

Train the Noise2Void U-Net with early stopping:

```bash
python src/train_n2v.py \
    --patch_csv data/manifests/patches_50_splits.csv \
    --outdir checkpoints/n2v_unet \
    --epochs 50 \
    --batch_size 8 \
    --lr 2e-4 \
    --base 48 \
    --hole 5 \
    --amp \
    --early_stop_patience 5 \
    --min_delta 1e-4 \
    --log_every 100
```

**Key Parameters:**
- `--base`: U-Net base channels (default: 48)
- `--hole`: Blind-spot mask size (default: 5×5)
- `--amp`: Enable automatic mixed precision
- `--early_stop_patience`: Stop after N epochs without improvement
- `--num_workers`: Data loading workers (use 0 on Windows, 4+ on Linux)

### 3. Inference

#### Preview Denoising
Generate side-by-side comparisons for test patches:

```bash
python src/tools/preview_denoise.py \
    --patch_csv data/manifests/patches_50_splits.csv \
    --ckpt checkpoints/n2v_unet/ckpt_3352.pt \
    --outdir reports/preview \
    --n 10
```

Output: `in_*.png` and `out_*.png` pairs in `reports/preview/`

#### Full Image Reconstruction
Reconstruct complete denoised images with cosine stitching:

```bash
python src/tools/denoise_full.py \
    --patch_csv data/manifests/patches_50_splits.csv \
    --parent_id 05b8fb844e0a \
    --ckpt checkpoints/n2v_unet/ckpt_3352.pt \
    --out reports/denoised_05b8fb844e0a.png
```

#### Interactive Analysis (GUI)
Run predictions with GUI file picker and generate detailed reports:

```bash
python src/predict_and_analyze.py
```

This will:
1. Open a file dialog to select an astronomical image
2. Denoise the image using the trained model
3. Detect stars in both original and denoised versions
4. Compute metrics (MAE, MSE, PSNR, SNR, brightness change)
5. Save cleaned image to `reports/predictions/cleaned_*.png`
6. Generate JSON report with full analysis
7. Display side-by-side visualization

### 4. Evaluation

#### Comprehensive Evaluation
Run full evaluation suite on test set:

```bash
python src/eval/metrics_and_plots.py \
    --patch_csv data/manifests/patches_50_splits.csv \
    --ckpt checkpoints/n2v_unet/ckpt_3352.pt \
    --out_dir reports/eval_quick \
    --use_cuda \
    --max_parents 20 \
    --fwhm 3.0 \
    --threshold_sigma 3.0
```

**Outputs:**
- `per_parent_metrics.csv`: PSNR, SSIM, MSE, MAE per image
- `psnr_hist.png`, `ssim_hist.png`: Metric distributions
- `psnr_vs_ssim.png`: Correlation plot
- `flux_scatter.png`: True vs denoised flux for matched sources
- `detection_confusion.png`: Detection confusion matrix
- `detection_summary.txt`: Precision, recall, TP/FP/FN counts

#### Detection PR Curves
Sweep detection thresholds to find optimal parameters:

```bash
python src/eval/detection_pr_curves.py \
    --patch_csv data/manifests/patches_50_splits.csv \
    --ckpt checkpoints/n2v_unet/ckpt_3352.pt \
    --out_dir reports/eval_detection \
    --use_cuda \
    --max_parents 20 \
    --thresholds "1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0"
```

**Outputs:**
- `pr_curve.png`: Precision-recall curve with threshold labels
- `prf_vs_threshold.png`: Precision/recall/F1 vs threshold
- `detection_counts_vs_threshold.png`: TP/FP/FN vs threshold
- `detection_threshold_sweep.csv`: Detailed metrics for each threshold
- `summary_metrics.txt`: Best F1 threshold and metrics

## Method Details

### Noise2Void Blind-Spot Training

The core idea is to train a denoising network without clean reference images:

1. **Blind-Spot Masking**: During training, random 5×5 pixel regions are masked (set to 0 in loss)
2. **Self-Supervision**: The network predicts the original noisy image, but only uses surrounding context (not the blind-spot itself)
3. **Loss Function**: Masked L2 loss computed only on unmasked regions
4. **Inference**: At test time, the full image is processed (no masking), producing denoised output

### U-Net Architecture

- **Encoder**: 4 downsampling blocks (MaxPool2d)
- **Bottleneck**: 16× base channels
- **Decoder**: 4 upsampling blocks (ConvTranspose2d) with skip connections
- **Channels**: 3 input/output (RGB), base=48 (default)
- **Activation**: SiLU (Swish)
- **Normalization**: BatchNorm2d

### Data Augmentation

Applied during training (astronomy-safe):
- Horizontal/vertical flips (50% probability each)
- Small rotations (±5°) with reflection padding
- Small translations (±8 pixels)
- Mild brightness jitter (±5%)

### Normalization

Percentile-based normalization (default: 1st-99.5th percentile) preserves faint features while clipping extreme outliers. Each patch stores normalization parameters for potential denormalization.

## Evaluation Metrics

### Pixel-Level
- **PSNR** (Peak Signal-to-Noise Ratio): Measures reconstruction quality
- **SSIM** (Structural Similarity Index): Perceptual similarity
- **MSE/MAE**: Mean squared/absolute error

### Astronomy-Specific
- **Star Detection**: DAOStarFinder with configurable FWHM and threshold
- **Source Matching**: Nearest-neighbor matching within max separation
- **Flux Preservation**: Scatter plots and residuals for matched sources
- **Detection Metrics**: Precision, recall, F1-score for source detection

## Checkpoints

Pre-trained checkpoints available in `checkpoints/n2v_unet/`:
- `ckpt_800.pt`: Early checkpoint (epoch ~5)
- `ckpt_1600.pt`: Mid-training (epoch ~10)
- `ckpt_3352.pt`: Final checkpoint (epoch ~20)

Load checkpoints:
```python
import torch
from models.unet_blindspot import UNetBlindspot

model = UNetBlindspot(in_ch=3, base=48)
ckpt = torch.load("checkpoints/n2v_unet/ckpt_3352.pt")
model.load_state_dict(ckpt["model"])
```

## Tips & Best Practices

### Training
- Start with `--num_workers 0` on Windows to avoid multiprocessing issues
- Use `--amp` for 2× speedup on modern GPUs (requires CUDA)
- Monitor validation loss; early stopping prevents overfitting
- Typical training: 10-20 epochs on ~1000 patches

### Data Preparation
- Use percentile normalization to handle wide dynamic range
- 512×512 patches work well; ensure sufficient overlap (stride=256)
- Keep parent-level splits to prevent data leakage

### Inference
- For large images, use `denoise_full.py` with cosine stitching
- GPU inference is 10-20× faster than CPU
- Denoised images preserve photometry better than median filtering

### Evaluation
- Use multiple detection thresholds to find optimal operating point
- Matched flux correlation should be >0.95 for good preservation
- Check PSNR >25 dB and SSIM >0.85 for acceptable quality

## Troubleshooting

**ImportError: photutils or astropy not found**
```bash
pip install photutils astropy scipy
```

**RuntimeError: CUDA out of memory**
- Reduce `--batch_size` (try 4 or 2)
- Use `--amp` for mixed precision
- Process fewer patches at once

**Windows multiprocessing errors**
- Set `--num_workers 0` in training

**Images too small for SSIM**
- SSIM requires images ≥7×7; code handles this gracefully with NaN

## Citation

If you use this code, please cite the Noise2Void paper:

```bibtex
@inproceedings{krull2019noise2void,
  title={Noise2void-learning denoising from single noisy images},
  author={Krull, Alexander and Buchholz, Tim-Oliver and Jug, Florian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2129--2137},
  year={2019}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Repository

**GitHub**: [gloooomed/astronomical-data-denoising](https://github.com/gloooomed/astronomical-data-denoising)

## Acknowledgments

- Noise2Void method by Krull et al. (2019)
- Photutils and Astropy communities
- PyTorch team for excellent deep learning framework