#!/usr/bin/env python3
"""
ESH-Vision Production Training Script
=======================================

NeurIPS-standard training pipeline with:
  • Mixup + CutMix augmentation
  • RandAugment transforms
  • PagedAdamW8bit (bitsandbytes) or AdamW fallback
  • Cosine LR decay with linear warmup
  • Ponder-cost warmup (0 until step 20k, then ramps to target)
  • Top-1 and Top-5 accuracy tracking
  • Resume from checkpoint
  • CSV logging for post-hoc training curves

Default config: Medium (embed_dim=256, depths=3,6,6) ≈ 150M params
Fits RTX 3080 Ti 12GB in ~66 hours for 15 epochs on ImageNet-100.

Usage
-----
    python train_final.py \\
        --hf_dataset clane9/imagenet-100 \\
        --epochs 15 \\
        --output_dir ./checkpoints
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import torchvision.transforms as transforms
    import torchvision.datasets as tv_datasets
    HAS_TV = True
except ImportError:
    HAS_TV = False

try:
    from datasets import load_dataset as hf_load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from esh_vision.model import ESHVisionBackbone, ESHVisionConfig


# ---------------------------------------------------------------------------
# HuggingFace dataset wrapper
# ---------------------------------------------------------------------------

class HFImageDataset(Dataset):
    """Wraps a HuggingFace image classification dataset as a PyTorch Dataset."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        features = hf_dataset.features
        if hasattr(features.get("label", None), "num_classes"):
            self.num_classes = features["label"].num_classes
        else:
            self.num_classes = len(set(hf_dataset["label"]))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        img = item["image"]
        label = item["label"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Mixup & CutMix augmentation (applied on GPU batches)
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=0.8):
    """Apply Mixup augmentation to a batch."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation to a batch."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = max(int(W * cut_ratio), 1)
    cut_h = max(int(H * cut_ratio), 1)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda for actual cut area
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def variance_incentive_loss(alphas: list[torch.Tensor]) -> torch.Tensor:
    """Negative variance of routing weights → encourages bimodal routing."""
    all_alpha = torch.cat([a.view(-1) for a in alphas])
    return -all_alpha.var()


def build_ponder_lambda(step: int, warmup_start: int = 20_000,
                        warmup_end: int = 25_000,
                        target: float = 0.01) -> float:
    """Ponder-cost coefficient: 0 until 20k steps, then ramps to target."""
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return target
    progress = (step - warmup_start) / (warmup_end - warmup_start)
    return target * progress


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def cosine_lr(optimizer, step: int, total_steps: int, lr: float,
              warmup_steps: int = 1000, min_lr: float = 1e-6):
    """Cosine LR with linear warmup."""
    if step < warmup_steps:
        lr_cur = lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr_cur = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr_cur
    return lr_cur


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = min(max(topk), output.shape[1])
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            k = min(k, output.shape[1])
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k / batch_size)
        return res


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """Simple CSV logger for training metrics."""

    def __init__(self, filepath: str, fieldnames: list[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.file = open(filepath, "w", newline="", buffering=1)
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row: dict):
        self.writer.writerow(row)

    def close(self):
        self.file.close()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ESH-Vision] Using device: {device}")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- Dataset -----------------------------------------------------------
    if args.hf_dataset:
        if not HAS_HF:
            sys.exit("Install `datasets` package: pip install datasets")
        if not HAS_TV:
            sys.exit("torchvision required for transforms.")

        print(f"[ESH-Vision] Loading HuggingFace dataset: {args.hf_dataset}")
        hf_kwargs = {"trust_remote_code": True}
        if args.hf_token:
            hf_kwargs["token"] = args.hf_token

        raw_ds = hf_load_dataset(args.hf_dataset, **hf_kwargs)

        split_name = args.hf_split
        if split_name not in raw_ds:
            available = list(raw_ds.keys())
            print(f"[ESH-Vision] Split '{split_name}' not found. Available: {available}")
            split_name = available[0]
            print(f"[ESH-Vision] Using split: '{split_name}'")

        # NeurIPS-standard augmentation pipeline
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        train_ds = HFImageDataset(raw_ds[split_name], transform=transform)
        num_classes = train_ds.num_classes
        print(f"[ESH-Vision] Dataset loaded: {len(train_ds)} samples, "
              f"{num_classes} classes")
    elif args.data_dir:
        if not HAS_TV:
            sys.exit("torchvision required for ImageFolder datasets.")
        num_classes = args.num_classes
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        train_ds = tv_datasets.ImageFolder(args.data_dir, transform=transform)
    else:
        sys.exit("Provide --hf_dataset or --data_dir.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
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
    num_params = model.get_num_params()
    print(f"[ESH-Vision] Parameters: {num_params:,}")

    # ---- Optimizer ---------------------------------------------------------
    if HAS_BNB and args.use_8bit:
        try:
            optimizer = bnb.optim.PagedAdamW8bit(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )
            print("[ESH-Vision] Using bitsandbytes PagedAdamW8bit")
        except AttributeError:
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )
            print("[ESH-Vision] Using bitsandbytes AdamW8bit (Paged not available)")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        print("[ESH-Vision] Using torch AdamW")

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ---- Resume from checkpoint --------------------------------------------
    start_epoch = 0
    global_step = 0
    best_acc = 0.0

    if args.resume and os.path.isfile(args.resume):
        print(f"[ESH-Vision] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"[ESH-Vision] Resumed at epoch {start_epoch}, step {global_step}, "
              f"best_acc={best_acc:.4f}")

    # ---- Logging -----------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "train_log.csv")
    csv_fields = [
        "epoch", "step", "loss", "ce_loss", "var_loss", "ponder_loss",
        "top1_acc", "top5_acc", "mean_alpha", "lr", "throughput", "time",
    ]
    # If resuming, append to existing log
    if args.resume and os.path.isfile(csv_path):
        logger_file = open(csv_path, "a", newline="", buffering=1)
        logger = csv.DictWriter(logger_file, fieldnames=csv_fields)
    else:
        logger_file = open(csv_path, "w", newline="", buffering=1)
        logger = csv.DictWriter(logger_file, fieldnames=csv_fields)
        logger.writeheader()

    # ---- Training ----------------------------------------------------------
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    effective_batch = args.batch_size * args.grad_accum_steps

    print(f"[ESH-Vision] Starting training for {args.epochs} epochs "
          f"({total_steps} steps)")
    print(f"[ESH-Vision] Batch size: {args.batch_size} x {args.grad_accum_steps} "
          f"accum = {effective_batch} effective")
    print(f"[ESH-Vision] Warmup: {args.warmup_epochs} epochs ({warmup_steps} steps)")
    print(f"[ESH-Vision] λ_var = {args.lambda_var}, "
          f"λ_ponder target = {args.lambda_ponder}")
    print(f"[ESH-Vision] Mixup α={args.mixup_alpha}, "
          f"CutMix α={args.cutmix_alpha}")

    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_top1 = 0.0
        epoch_top5 = 0.0
        epoch_alpha = 0.0
        n_logged = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    dynamic_ncols=True)
        batch_start = time.time()

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # --- Mixup / CutMix (50/50 random choice) ----------------------
            use_mixup = random.random() < 0.5
            if args.mixup_alpha > 0 and use_mixup:
                images, targets_a, targets_b, lam = mixup_data(
                    images, labels, alpha=args.mixup_alpha)
            elif args.cutmix_alpha > 0 and not use_mixup:
                images, targets_a, targets_b, lam = cutmix_data(
                    images, labels, alpha=args.cutmix_alpha)
            else:
                targets_a, targets_b, lam = labels, labels, 1.0

            # --- Forward pass -----------------------------------------------
            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(images)
                logits = out["features"]
                alphas = out["alphas"]
                ponder_cost = out["ponder_cost"]

                # Classification loss (Mixup-aware)
                ce = mixup_criterion(ce_loss_fn, logits, targets_a, targets_b, lam)

                # Variance incentive loss
                var_loss = variance_incentive_loss(alphas)

                # Ponder cost with warmup
                lam_ponder = build_ponder_lambda(
                    global_step,
                    warmup_start=args.ponder_warmup_start,
                    warmup_end=args.ponder_warmup_end,
                    target=args.lambda_ponder,
                )

                total_loss = (
                    ce
                    + args.lambda_var * var_loss
                    + lam_ponder * ponder_cost
                )
                total_loss = total_loss / args.grad_accum_steps

            # --- Backward ---------------------------------------------------
            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # --- LR schedule -----------------------------------------------
            lr_cur = cosine_lr(
                optimizer, global_step, total_steps,
                args.lr, warmup_steps=warmup_steps,
            )
            global_step += 1

            # --- Metrics ----------------------------------------------------
            with torch.no_grad():
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                mean_alpha = torch.cat([a.view(-1) for a in alphas]).mean().item()

            elapsed = time.time() - batch_start
            throughput = args.batch_size / max(elapsed, 1e-6)
            batch_start = time.time()

            epoch_loss += total_loss.item() * args.grad_accum_steps
            epoch_top1 += top1.item()
            epoch_top5 += top5.item()
            epoch_alpha += mean_alpha
            n_logged += 1

            # --- Progress bar -----------------------------------------------
            if batch_idx % args.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{total_loss.item() * args.grad_accum_steps:.4f}",
                    "top1": f"{top1.item():.3f}",
                    "top5": f"{top5.item():.3f}",
                    "α": f"{mean_alpha:.3f}",
                    "lr": f"{lr_cur:.1e}",
                    "img/s": f"{throughput:.1f}",
                })

            # --- CSV log every N steps -------------------------------------
            if global_step % args.log_every == 0:
                logger.writerow({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": f"{total_loss.item() * args.grad_accum_steps:.6f}",
                    "ce_loss": f"{ce.item():.6f}",
                    "var_loss": f"{var_loss.item():.6f}",
                    "ponder_loss": f"{ponder_cost.item():.6f}",
                    "top1_acc": f"{top1.item():.4f}",
                    "top5_acc": f"{top5.item():.4f}",
                    "mean_alpha": f"{mean_alpha:.4f}",
                    "lr": f"{lr_cur:.8f}",
                    "throughput": f"{throughput:.2f}",
                    "time": f"{time.time():.0f}",
                })

        # --- End of epoch stats ---------------------------------------------
        avg_loss = epoch_loss / max(n_logged, 1)
        avg_top1 = epoch_top1 / max(n_logged, 1)
        avg_top5 = epoch_top5 / max(n_logged, 1)
        avg_alpha = epoch_alpha / max(n_logged, 1)

        print(f"\n[ESH-Vision] Epoch {epoch+1}/{args.epochs} — "
              f"loss={avg_loss:.4f}, top1={avg_top1:.4f}, "
              f"top5={avg_top5:.4f}, α={avg_alpha:.3f}")

        # --- Checkpoint -----------------------------------------------------
        is_best = avg_top1 > best_acc
        best_acc = max(avg_top1, best_acc)

        if (epoch + 1) % args.save_every == 0 or is_best:
            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config.__dict__,
                "best_acc": best_acc,
                "args": vars(args),
            }
            ckpt_path = os.path.join(
                args.output_dir, f"esh_vision_ep{epoch+1}.pt")
            try:
                torch.save(ckpt, ckpt_path)
                print(f"[ESH-Vision] Saved checkpoint: {ckpt_path}")
            except OSError as e:
                print(f"[ESH-Vision] WARNING: Could not save checkpoint: {e}")

            if is_best:
                best_path = os.path.join(args.output_dir, "esh_vision_best.pt")
                try:
                    torch.save(ckpt, best_path)
                    print(f"[ESH-Vision] New best! top1={best_acc:.4f} → {best_path}")
                except OSError as e:
                    print(f"[ESH-Vision] WARNING: Could not save best: {e}")

    logger_file.close()
    print(f"\n[ESH-Vision] Training complete. Best top-1: {best_acc:.4f}")
    print(f"[ESH-Vision] Log saved to: {csv_path}")


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ESH-Vision Production Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--hf_dataset", type=str, default="clane9/imagenet-100",
                    help="HuggingFace dataset name")
    g.add_argument("--hf_split", type=str, default="train")
    g.add_argument("--hf_token", type=str, default=None)
    g.add_argument("--data_dir", type=str, default=None,
                    help="Local ImageFolder directory (alternative to HF)")
    g.add_argument("--num_classes", type=int, default=100)
    g.add_argument("--img_size", type=int, default=224)
    g.add_argument("--patch_size", type=int, default=16)

    # Model architecture (Medium defaults for 70-hour budget)
    g = p.add_argument_group("Model")
    g.add_argument("--embed_dim", type=int, default=256)
    g.add_argument("--depths", type=str, default="3,6,6")
    g.add_argument("--num_heads", type=str, default="4,8,8")
    g.add_argument("--ssm_d_state", type=int, default=16)
    g.add_argument("--act_threshold", type=float, default=0.7)
    g.add_argument("--max_ponder", type=int, default=2)
    g.add_argument("--drop_rate", type=float, default=0.1)

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=15)
    g.add_argument("--warmup_epochs", type=int, default=2,
                    help="Linear warmup epochs")
    g.add_argument("--batch_size", type=int, default=16)
    g.add_argument("--grad_accum_steps", type=int, default=8)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--weight_decay", type=float, default=0.05)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=42)

    # Augmentation
    g = p.add_argument_group("Augmentation")
    g.add_argument("--mixup_alpha", type=float, default=0.8,
                    help="Mixup alpha (0 to disable)")
    g.add_argument("--cutmix_alpha", type=float, default=1.0,
                    help="CutMix alpha (0 to disable)")

    # Losses
    g = p.add_argument_group("Losses")
    g.add_argument("--lambda_var", type=float, default=0.1)
    g.add_argument("--lambda_ponder", type=float, default=0.01)
    g.add_argument("--ponder_warmup_start", type=int, default=20000)
    g.add_argument("--ponder_warmup_end", type=int, default=25000)

    # Hardware
    g = p.add_argument_group("Hardware")
    g.add_argument("--amp", action="store_true", default=True)
    g.add_argument("--no_amp", action="store_true")
    g.add_argument("--use_8bit", action="store_true", default=True)
    g.add_argument("--use_checkpoint", action="store_true", default=True)
    g.add_argument("--num_workers", type=int, default=4)

    # Logging & checkpointing
    g = p.add_argument_group("Logging")
    g.add_argument("--log_every", type=int, default=10)
    g.add_argument("--save_every", type=int, default=3)
    g.add_argument("--output_dir", type=str, default="./checkpoints")
    g.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")

    args = p.parse_args()
    if args.no_amp:
        args.amp = False
    return args


if __name__ == "__main__":
    train(parse_args())
