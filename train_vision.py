#!/usr/bin/env python3
"""
ESH-Vision Training Script
===========================

Supports ImageNet-style datasets with:
  • Cross-entropy + Variance-Incentive loss + Ponder-cost loss
  • AMP (Automatic Mixed Precision)
  • 8-bit AdamW via bitsandbytes
  • Gradient checkpointing (enabled by default)
  • Ponder-cost warmup scheduler (0 → target after 15k steps)
  • Cosine LR schedule with warmup

Usage
-----
    python train_vision.py \\
        --data_dir /path/to/imagenet \\
        --epochs 90 \\
        --batch_size 32 \\
        --lr 1e-3 \\
        --num_classes 1000

For a quick smoke test with random data::

    python train_vision.py --smoke_test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from esh_vision.model import ESHVisionBackbone, ESHVisionConfig

# Optional heavy imports — guarded so smoke test works without them
try:
    import bitsandbytes as bnb

    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    from torchvision import datasets, transforms

    HAS_TV = True
except ImportError:
    HAS_TV = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # noqa: D103
        return iterable


# ---------------------------------------------------------------------------
# Random dummy dataset for smoke testing
# ---------------------------------------------------------------------------

class RandomImageDataset(Dataset):
    """Generates random images + labels.  For smoke / unit tests only."""

    def __init__(self, num_samples: int = 256, img_size: int = 224,
                 num_classes: int = 10):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        img = torch.rand(3, self.img_size, self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def variance_incentive_loss(alphas: list[torch.Tensor]) -> torch.Tensor:
    """Negative variance of routing weights → encourages bimodal distribution.

    L_var = -Var(α)  (we *subtract* this from total loss, so minimising
    total loss *maximises* variance → decisive routing).
    """
    all_alpha = torch.cat([a.view(-1) for a in alphas])
    return -all_alpha.var()


def build_ponder_lambda(step: int, warmup_start: int = 15_000,
                        warmup_end: int = 20_000,
                        target: float = 0.01) -> float:
    """Ponder-cost coefficient with warmup.

    Returns 0 before ``warmup_start``, linearly ramps to ``target``
    between ``warmup_start`` and ``warmup_end``, then stays at ``target``.
    """
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return target
    progress = (step - warmup_start) / (warmup_end - warmup_start)
    return target * progress


# ---------------------------------------------------------------------------
# LR scheduler helpers
# ---------------------------------------------------------------------------

def cosine_lr(optimizer, step: int, total_steps: int, lr: float,
              warmup_steps: int = 1000, min_lr: float = 1e-6):
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        lr_cur = lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr_cur = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr_cur
    return lr_cur


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ESH-Vision] Using device: {device}")

    # ---- Dataset -----------------------------------------------------------
    if args.smoke_test:
        print("[ESH-Vision] Smoke test mode — using random data")
        num_classes = 10
        train_ds = RandomImageDataset(
            num_samples=args.batch_size * 4,
            img_size=args.img_size,
            num_classes=num_classes,
        )
    else:
        if not HAS_TV:
            sys.exit("torchvision is required for real datasets. "
                     "Install it or use --smoke_test.")
        num_classes = args.num_classes
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
        ])
        train_ds = datasets.ImageFolder(args.data_dir, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Model -------------------------------------------------------------
    config = ESHVisionConfig(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depths=[int(d) for d in args.depths.split(",")],
        num_heads=[int(h) for h in args.num_heads.split(",")],
        ssm_d_state=args.ssm_d_state,
        act_threshold=args.act_threshold,
        max_ponder=args.max_ponder,
        drop_rate=args.drop_rate,
        use_checkpoint=args.use_checkpoint,
        num_classes=num_classes,
    )
    model = ESHVisionBackbone(config).to(device)
    print(f"[ESH-Vision] Parameters: {model.get_num_params():,}")

    # ---- Optimizer ---------------------------------------------------------
    if HAS_BNB and args.use_8bit:
        print("[ESH-Vision] Using bitsandbytes 8-bit AdamW")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        if args.use_8bit:
            print("[ESH-Vision] bitsandbytes not found — falling back to torch AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )

    # ---- AMP scaler --------------------------------------------------------
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    # ---- Criterion ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # ---- Training ----------------------------------------------------------
    total_steps = len(train_loader) * args.epochs
    global_step = 0
    log_data: list[dict] = []

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[ESH-Vision] Starting training for {args.epochs} epochs "
          f"({total_steps} steps)")
    print(f"[ESH-Vision] Batch size: {args.batch_size} x {args.grad_accum_steps} accum "
          f"= {args.batch_size * args.grad_accum_steps} effective")
    print(f"[ESH-Vision] λ_var = {args.lambda_var}, "
          f"λ_ponder target = {args.lambda_ponder}")

    accum_steps = args.grad_accum_steps

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # LR schedule (update per optimizer step, not per micro-batch)
            if batch_idx % accum_steps == 0:
                lr_cur = cosine_lr(
                    optimizer, global_step, total_steps // accum_steps, args.lr,
                    warmup_steps=args.warmup_steps,
                )

            # Ponder cost coefficient
            lambda_p = build_ponder_lambda(
                global_step,
                warmup_start=args.ponder_warmup_start,
                warmup_end=args.ponder_warmup_end,
                target=args.lambda_ponder,
            )

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                out = model(images)
                logits = out["features"]
                alphas = out["alphas"]
                ponder_cost = out["ponder_cost"]

                # Cross-entropy
                ce_loss = criterion(logits, labels)

                # Variance incentive: -λ_var * Var(α)
                var_loss = args.lambda_var * variance_incentive_loss(alphas)

                # Ponder cost
                pond_loss = lambda_p * ponder_cost

                total_loss = (ce_loss + var_loss + pond_loss) / accum_steps

            scaler.scale(total_loss).backward()

            # Optimizer step every accum_steps micro-batches
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # Metrics
            acc = (logits.argmax(-1) == labels).float().mean().item()
            epoch_loss += total_loss.item() * accum_steps
            epoch_acc += acc
            num_batches += 1

            if batch_idx % args.log_every == 0:
                mean_alpha = torch.cat(
                    [a.view(-1) for a in alphas]
                ).mean().item()
                info = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "lr": lr_cur,
                    "ce_loss": ce_loss.item(),
                    "var_loss": var_loss.item(),
                    "ponder_loss": pond_loss.item(),
                    "total_loss": total_loss.item(),
                    "acc": acc,
                    "mean_alpha": mean_alpha,
                    "lambda_ponder": lambda_p,
                }
                log_data.append(info)
                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    acc=f"{acc:.3f}",
                    α=f"{mean_alpha:.3f}",
                    lr=f"{lr_cur:.2e}",
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        print(f"  → Epoch {epoch+1}: loss={avg_loss:.4f}  acc={avg_acc:.3f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = Path(args.output_dir) / f"esh_vision_ep{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    # Save training log
    log_path = Path(args.output_dir) / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"[ESH-Vision] Training log saved: {log_path}")
    print("[ESH-Vision] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ESH-Vision Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data_dir", type=str, default="./data/imagenet",
                    help="Path to ImageFolder-style dataset")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=4)

    # Model
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depths", type=str, default="4,8,10",
                    help="Comma-separated block depths per stage")
    p.add_argument("--num_heads", type=str, default="6,12,12",
                    help="Comma-separated attention heads per stage")
    p.add_argument("--ssm_d_state", type=int, default=16)
    p.add_argument("--act_threshold", type=float, default=0.7)
    p.add_argument("--max_ponder", type=int, default=2)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--use_checkpoint", action="store_true", default=True,
                    help="Enable gradient checkpointing")
    p.add_argument("--no_checkpoint", dest="use_checkpoint",
                    action="store_false")

    # Training
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=8,
                    help="Gradient accumulation steps (effective batch = batch_size * this)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Losses
    p.add_argument("--lambda_var", type=float, default=0.1,
                    help="Coefficient for variance-incentive loss")
    p.add_argument("--lambda_ponder", type=float, default=0.01,
                    help="Target ponder-cost coefficient (after warmup)")
    p.add_argument("--ponder_warmup_start", type=int, default=15000,
                    help="Step to begin ponder-cost warmup")
    p.add_argument("--ponder_warmup_end", type=int, default=20000,
                    help="Step at which ponder-cost reaches target")

    # Hardware
    p.add_argument("--amp", action="store_true", default=True,
                    help="Use Automatic Mixed Precision")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--use_8bit", action="store_true", default=True,
                    help="Use bitsandbytes 8-bit AdamW")
    p.add_argument("--no_8bit", dest="use_8bit", action="store_false")

    # Output
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=10)

    # Smoke test
    p.add_argument("--smoke_test", action="store_true",
                    help="Run a quick smoke test with random data")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
