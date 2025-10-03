# src/train_n2v.py
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

# ---- AMP imports (compat for old/new PyTorch) ----
try:
    from torch.amp import autocast, GradScaler  # PyTorch 2.x+
    HAS_TORCH_AMP = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler  # PyTorch < 2.0
    HAS_TORCH_AMP = False

from dataio.loaders import make_loaders
from models.unet_blindspot import UNetBlindspot
from losses.masked_loss import make_center_mask, masked_l2


def save_ckpt(model, opt, step, outdir: str | Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
        outdir / f"ckpt_{step}.pt",
    )


def train(args):
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    if use_cuda:
        torch.backends.cudnn.benchmark = True  # faster on fixed 512x512

    train_loader, val_loader, _ = make_loaders(
        Path(args.patch_csv),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,   # only pin on CUDA
        use_aug=True,
    )

    model = UNetBlindspot(in_ch=3, base=args.base).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- AMP setup (PyTorch 2.6 prefers positional device arg) ----
    if HAS_TORCH_AMP:
        scaler = GradScaler("cuda" if use_cuda else "cpu",
                            enabled=(args.amp and use_cuda))
        autocast_args = dict(device_type=("cuda" if use_cuda else "cpu"),
                             enabled=(args.amp and use_cuda))
    else:
        scaler = GradScaler(enabled=(args.amp and use_cuda))
        autocast_args = dict(enabled=(args.amp and use_cuda))

    global_step = 0
    best_val = float("inf")
    no_improve = 0  # for early stopping

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for it, batch in enumerate(train_loader, 1):
            x = batch["image"].to(device, non_blocking=use_cuda)  # [B,3,512,512]
            B, C, H, W = x.shape

            # Blind-spot mask: 0 = hidden region, 1 = contributes to loss
            mask = make_center_mask(B, H, W, hole=args.hole, device=device)
            with torch.no_grad():
                target = x  # predict x itself; masked region excluded from loss

            opt.zero_grad(set_to_none=True)
            with autocast(**autocast_args):
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

            # Limit work per epoch if requested
            if args.max_steps_per_epoch and it >= args.max_steps_per_epoch:
                break

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            vcount = 0
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=use_cuda)
                B, C, H, W = x.shape
                mask = make_center_mask(B, H, W, hole=args.hole, device=device)
                with autocast(**autocast_args):
                    y = model(x)
                    vloss += masked_l2(y, x, mask).item()
                vcount += 1
            vloss /= max(1, vcount)

        print(f"[val] epoch {epoch} masked-L2 {vloss:.5f}")

        # ---- early stopping + best checkpoint ----
        if vloss + args.min_delta < best_val:
            best_val = vloss
            no_improve = 0
            save_ckpt(model, opt, global_step, args.outdir)
        else:
            no_improve += 1
            print(f"[val] no improvement count = {no_improve}")
            if no_improve >= args.early_stop_patience:
                print(f"[early stop] stopping after {args.early_stop_patience} epochs without improvement")
                break

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

    # New controls
    ap.add_argument("--max_steps_per_epoch", type=int, default=0,
                    help="If >0, limit number of training batches per epoch.")
    ap.add_argument("--early_stop_patience", type=int, default=3,
                    help="Stop after N epochs with no val improvement.")
    ap.add_argument("--min_delta", type=float, default=1e-4,
                    help="Minimum improvement in val loss to count as progress.")

    args = ap.parse_args()
    train(args)
