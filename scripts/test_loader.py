from pathlib import Path
import torch
from dataio.loaders import make_loaders

if __name__ == "__main__":

    train_loader, val_loader, test_loader = make_loaders(
        Path(r"data/manifests/patches_splits.csv"),
        batch_size=8, num_workers=0, pin_memory=False, use_aug=False
    )

    batch = next(iter(train_loader))
    x = batch["image"]  # [B, C, 512, 512]
    print("Train batch:", x.shape, x.dtype, float(x.min()), float(x.max()))
    assert x.shape[1:] == (3, 512, 512)
    print("OK")
