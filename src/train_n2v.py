from __future__ import annotations
import os, math, time, argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from dataio.loaders import make_loaders
from models.unet_blindspot import UNetBlindspot
from losses.masked_loss import make_center_mask, masked_l2

def save_ckpt(model, opt, step, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, outdir / f"ckpt_{step}.pt")

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _ = make_loaders(
        Path(args.patch_csv), batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, use_aug=True
    )

    model = UNetBlindspot(in_ch=3, base=args.base).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    global_step = 0
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for it, batch in enumerate(train_loader, 1):
            x = batch["image"].to(device, non_blocking=True)  # [B,3,512,512]
            B, C, H, W = x.shape

            # Build blind-spot mask (0 where we hide the center; 1 elsewhere)
            mask = make_center_mask(B, H, W, hole=args.hole, device=device)
            with torch.no_grad():
                target = x.clone()
                # Optional: we can optionally inpaint the masked region with noise to discourage copying
                # target[:, :, mask.repeat(1, C, 1, 1) == 0] = 0.0

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                y = model(x)
                loss = masked_l2(y, target, mask)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                print(f"[train] epoch {epoch} step {global_step} loss {running/args.log_every:.5f}")
                running = 0.0

        # --- validation ---
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            vcount = 0
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                B, C, H, W = x.shape
                mask = make_center_mask(B, H, W, hole=args.hole, device=device)
                y = model(x)
                vloss += masked_l2(y, x, mask).item()
                vcount += 1
            vloss /= max(1, vcount)
        print(f"[val] epoch {epoch} masked-L2 {vloss:.5f}")

        # save best
        if vloss < best_val:
            best_val = vloss
            save_ckpt(model, opt, global_step, args.outdir)

    print(f"[done] best val masked-L2: {best_val:.5f}  ckpts in {args.outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_csv", type=str, default="data/manifests/patches_splits.csv")
    ap.add_argument("--outdir", type=str, default="checkpoints/n2v_unet")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)  # Windows-safe; bump on Linux/Colab
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--hole", type=int, default=5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    train(args)
