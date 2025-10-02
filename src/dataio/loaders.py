from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---- Simple, astronomy-safe augmentations on tensors (C,H,W)
def build_augment_pipeline(
    hflip: bool=True, vflip: bool=True, rot_deg: int=5, translate_px: int=8, brightness: float=0.05
) -> Callable[[torch.Tensor], torch.Tensor]:
    import torch.nn.functional as F
    import math
    gen = torch.Generator()

    def _aug(x: torch.Tensor) -> torch.Tensor:
        # x: C,H,W in [0,1]
        C, H, W = x.shape

        # flips
        if hflip and torch.rand((), generator=gen) < 0.5:
            x = torch.flip(x, dims=[2])
        if vflip and torch.rand((), generator=gen) < 0.5:
            x = torch.flip(x, dims=[1])

        # small rotation + translation (affine)
        angle = (torch.rand((), generator=gen)*2-1) * rot_deg  # [-rot_deg, rot_deg]
        tx = (torch.rand((), generator=gen)*2-1) * translate_px
        ty = (torch.rand((), generator=gen)*2-1) * translate_px

        # build affine grid
        theta = torch.zeros(1,2,3, dtype=x.dtype, device=x.device)
        a = math.radians(float(angle))
        cos, sin = math.cos(a), math.sin(a)
        theta[0,0,0] = cos; theta[0,0,1] = -sin; theta[0,0,2] = (2*tx)/(W-1)
        theta[0,1,0] = sin; theta[0,1,1] =  cos; theta[0,1,2] = (2*ty)/(H-1)

        grid = F.affine_grid(theta, size=(1,C,H,W), align_corners=False)
        x = F.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)

        # mild brightness jitter (preserve flux roughly)
        if brightness > 0:
            delta = (torch.rand((), generator=gen)*2-1) * brightness
            x = torch.clamp(x + delta, 0.0, 1.0)
        return x
    return _aug

class PatchDataset(Dataset):
    def __init__(self, patch_csv: Path, split: str, transforms: Optional[Callable]=None):
        df = pd.read_csv(patch_csv)
        if "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        arr = np.load(row["path"]).astype(np.float32)  # H,W,C in [0,1]
        arr = np.transpose(arr, (2, 0, 1))            # C,H,W
        x = torch.from_numpy(arr)

        if self.transforms is not None:
            x = self.transforms(x)

        return {
            "image": x,  # C,H,W
            "parent_id": row["parent_id"],
            "tile_id": row["tile_id"],
            "yx": (int(row["y"]), int(row["x"]))
        }

def make_loaders(
    patch_csv: Path,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_aug: bool = True
):
    aug = build_augment_pipeline() if use_aug else None
    train_ds = PatchDataset(patch_csv, split="train", transforms=aug)
    val_ds   = PatchDataset(patch_csv, split="val",   transforms=None)
    test_ds  = PatchDataset(patch_csv, split="test",  transforms=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
