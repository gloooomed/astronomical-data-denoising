# Astronomical Image Denoising

Self-supervised denoising of astronomical images using Noise2Void with blind-spot U-Net architecture.

## Overview

This project implements self-supervised deep learning for astronomical image denoising using the Noise2Void method. The system does not require clean reference images for training and includes tools for data preparation, model training, inference, and evaluation.

## Features

- Self-supervised learning without clean reference images
- Blind-spot training with 5×5 pixel masking
- U-Net architecture with skip connections
- Early stopping and mixed precision training
- Percentile-based normalization for astronomical data
- Overlapping tile extraction and cosine-weighted stitching
- Star detection and flux preservation analysis
- Interactive GUI for prediction and analysis

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
- CUDA-capable GPU (optional)

### Setup
```bash
git clone https://github.com/gloooomed/astronomical-data-denoising.git
cd astronomical-data-denoising
pip install -r requirements.txt
```

### Core Dependencies
```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=8.0.0
tqdm>=4.60.0
photutils
astropy
scikit-image
matplotlib
```

## Usage

### Training

```bash
python src/train_n2v.py \
    --patch_csv data/manifests/patches_50_splits.csv \
    --outdir checkpoints/n2v_unet \
    --epochs 50 \
    --batch_size 8 \
    --lr 2e-4 \
    --base 48 \
    --amp
```

### Inference

#### GUI Application
```bash
python src/predict_and_analyze_gui.py
```

#### Command Line
```bash
python src/predict_and_analyze.py --image path/to/image.png
```

### Evaluation

```bash
python src/eval/metrics_and_plots.py \
    --patch_csv data/manifests/patches_50_splits.csv \
    --ckpt checkpoints/n2v_unet/ckpt_3352.pt \
    --out_dir reports/eval_quick
```

## Method

### Noise2Void

Noise2Void enables self-supervised denoising without clean reference images. During training, random pixel regions are masked and the network learns to predict pixel values using only surrounding context. At inference, the full image is processed to produce denoised output.

### Architecture

- U-Net with 4 encoder and 4 decoder blocks
- Skip connections between encoder and decoder
- Base channels: 48 (configurable)
- Activation: SiLU
- Normalization: BatchNorm2d

### Training

- Blind-spot masking: 5×5 pixel regions
- Loss: Masked L2 on unmasked pixels
- Optimizer: AdamW
- Mixed precision training supported
- Early stopping based on validation loss

## Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MSE/MAE (Mean Squared/Absolute Error)
- Star detection accuracy (DAOStarFinder)
- Flux preservation analysis
- Precision, recall, and F1-score

## Citation

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

MIT License