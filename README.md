# Astronomical Data Denoising

Self-supervised denoising of astronomical images using Noise2Void with blind-spot U-Net training.

## Structure

- `src/train_n2v.py` - Training script
- `src/models/unet_blindspot.py` - U-Net architecture  
- `src/losses/masked_loss.py` - Blind-spot masking and loss
- `src/dataio/` - Data loading and preprocessing
- `data/patches/` - 512x512 image patches (.npy files)
- `data/manifests/` - Metadata and train/val/test splits

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/train_n2v.py --patch_csv data/manifests/patches_splits.csv
```

## Method

Uses blind-spot training where random 5x5 pixel regions are masked during training, forcing the network to learn denoising from surrounding context without requiring clean reference images.