# src/utils/stitch.py
from __future__ import annotations
import numpy as np

def cosine_window_2d(h, w):
    wy = 0.5 * (1 - np.cos(2*np.pi*np.arange(h)/(h-1)))
    wx = 0.5 * (1 - np.cos(2*np.pi*np.arange(w)/(w-1)))
    return np.outer(wy, wx).astype(np.float32)

def stitch_tiles(tiles, coords, out_shape):
    H, W, C = out_shape
    acc = np.zeros(out_shape, dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    h, w, _ = tiles[0].shape
    win = cosine_window_2d(h, w)[:, :, None]

    for tile, (y, x) in zip(tiles, coords):
        acc[y:y+h, x:x+w, :] += tile * win
        wsum[y:y+h, x:x+w, :] += win

    wsum = np.clip(wsum, 1e-6, None)
    return np.clip(acc / wsum, 0.0, 1.0)
