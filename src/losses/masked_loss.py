from __future__ import annotations
import torch
import torch.nn.functional as F

def make_center_mask(B, H, W, hole=5, device=None):
    """Randomly choose center positions and zero their contribution to loss within a (hole x hole) box."""
    device = device or "cpu"
    mask = torch.ones((B, 1, H, W), device=device, dtype=torch.float32)
    ys = torch.randint(low=hole//2, high=H - hole//2, size=(B,), device=device)
    xs = torch.randint(low=hole//2, high=W - hole//2, size=(B,), device=device)
    for b in range(B):
        y0, y1 = ys[b] - hole//2, ys[b] + hole//2 + (hole % 2 == 0)
        x0, x1 = xs[b] - hole//2, xs[b] + hole//2 + (hole % 2 == 0)
        mask[b, :, y0:y1, x0:x1] = 0.0
    return mask  # 1 in loss area, 0 in blind-spot

def masked_l2(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff.mean(dim=1, keepdim=True)  # average across channels
    num = (mask * diff).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den
