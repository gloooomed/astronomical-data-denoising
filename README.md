# Astronomical Data Denoising

Self-supervised denoising of astronomical images using Noise2Void with blind-spot U-Net training.

## Features

- **Advanced Training**: Early stopping, AMP support, PyTorch 2.x compatibility
- **Complete Pipeline**: Data manifest creation, tiling, splitting, and QC tools
- **Inference Tools**: Single patch preview and full image reconstruction
- **Smart Stitching**: Cosine windowing for seamless tile reconstruction

## Structure

```
src/
├── train_n2v.py              # Enhanced training with early stopping
├── models/unet_blindspot.py   # U-Net architecture
├── losses/masked_loss.py      # Blind-spot masking and loss
├── utils/stitch.py           # Tile stitching with cosine windows
├── tools/
│   ├── preview_denoise.py    # Preview denoising on test patches
│   └── denoise_full.py       # Reconstruct full denoised images
└── dataio/
    ├── manifest.py           # Build image catalogs with metadata
    ├── tiler.py             # Extract overlapping patches
    ├── split.py             # Create train/val/test splits
    ├── qc.py                # Quality control validation
    └── loaders.py           # Data loading with augmentations
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/train_n2v.py --patch_csv data/manifests/patches_splits.csv --epochs 50 --early_stop_patience 5
```

### Preview Results
```bash
python src/tools/preview_denoise.py --ckpt checkpoints/n2v_unet/ckpt_best.pt --n 10
```

### Full Image Reconstruction
```bash
python src/tools/denoise_full.py --patch_csv data/manifests/patches_splits.csv --parent_id fd2582f57237 --ckpt checkpoints/n2v_unet/ckpt_best.pt --out denoised_full.png
```

## Method

Uses blind-spot training where random 5x5 pixel regions are masked during training, forcing the network to learn denoising from surrounding context without requiring clean reference images.