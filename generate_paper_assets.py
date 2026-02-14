#!/usr/bin/env python3
"""
ESH-Vision Paper Asset Generator
==================================

Generates publication-ready figures from a trained checkpoint:
  1. α-Saliency Heatmaps — overlay routing intensity on input images
  2. Entropy vs α Scatter Plot — prove "Elastic Intelligence" correlation
  3. Training Curves — loss, accuracy, α from CSV logs

Usage
-----
    python generate_paper_assets.py \\
        --checkpoint checkpoints/esh_vision_best.pt \\
        --hf_dataset clane9/imagenet-100 \\
        --output_dir paper_assets

    # Just plot training curves from log:
    python generate_paper_assets.py \\
        --log_csv checkpoints/train_log.csv \\
        --output_dir paper_assets \\
        --curves_only
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torchvision.transforms as transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False

try:
    from datasets import load_dataset as hf_load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False

from esh_vision.model import ESHVisionBackbone, ESHVisionConfig


# ---------------------------------------------------------------------------
# Saliency heatmaps
# ---------------------------------------------------------------------------

def generate_saliency_heatmaps(model, images, save_dir, device, num_images=16):
    """Generate α-value heatmaps overlaid on input images.

    For each layer, we extract the routing coefficient α and overlay it
    as a heatmap (red=high α → Attention, blue=low α → VSSM).
    """
    if not HAS_MPL or not HAS_PIL:
        print("[WARNING] matplotlib and Pillow required for heatmaps.")
        return

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for img_idx in range(min(num_images, len(images))):
        img_tensor = images[img_idx].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor)

        alphas = out["alphas"]  # list of (1, N, 1) per block

        # Denormalize image for display
        img_display = images[img_idx].cpu() * std + mean
        img_display = img_display.clamp(0, 1).permute(1, 2, 0).numpy()

        # Group alphas by stage
        config = model.config
        depths = config.depths
        stage_alphas = []
        idx = 0
        for stage_idx, depth in enumerate(depths):
            stage_a = []
            for _ in range(depth):
                if idx < len(alphas):
                    stage_a.append(alphas[idx].squeeze(-1).squeeze(0).cpu())  # (N,)
                    idx += 1
            if stage_a:
                # Average α across blocks in this stage
                avg_alpha = torch.stack(stage_a).mean(0)  # (N,)
                stage_alphas.append((stage_idx, avg_alpha))

        # Create figure: original + one heatmap per stage
        n_stages = len(stage_alphas)
        fig, axes = plt.subplots(1, n_stages + 1, figsize=(4 * (n_stages + 1), 4))

        if n_stages + 1 == 1:
            axes = [axes]

        axes[0].imshow(img_display)
        axes[0].set_title("Input Image", fontsize=10)
        axes[0].axis("off")

        for i, (stage_idx, alpha_flat) in enumerate(stage_alphas):
            grid_side = int(math.sqrt(len(alpha_flat)))
            alpha_map = alpha_flat.view(grid_side, grid_side).numpy()

            # Upsample to image size for overlay
            alpha_up = np.array(Image.fromarray(alpha_map).resize(
                (img_display.shape[1], img_display.shape[0]),
                Image.BILINEAR,
            ))

            axes[i + 1].imshow(img_display)
            im = axes[i + 1].imshow(alpha_up, cmap="RdBu_r",
                                     alpha=0.6, vmin=0, vmax=1)
            axes[i + 1].set_title(f"Stage {stage_idx+1} α\n"
                                   f"(mean={alpha_flat.mean():.3f})",
                                   fontsize=10)
            axes[i + 1].axis("off")

        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04,
                     label="α (→1: Attention, →0: VSSM)")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"saliency_{img_idx:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Entropy vs α scatter plot
# ---------------------------------------------------------------------------

def generate_entropy_scatter(model, images, save_dir, device, num_images=50):
    """Scatter plot of per-patch pixel-entropy vs routing α.

    If entropy and α are correlated, it proves the "Elastic Intelligence"
    hypothesis: complex patches → Attention, simple patches → VSSM.
    """
    if not HAS_MPL:
        print("[WARNING] matplotlib required for scatter plot.")
        return

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_entropy = []
    all_alpha = []

    for img_idx in range(min(num_images, len(images))):
        img_tensor = images[img_idx].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor)

        alphas = out["alphas"]
        entropy_maps = out.get("entropy")

        if entropy_maps is None:
            print("[WARNING] Model did not return entropy values. "
                  "Check model.forward() returns 'entropy' key.")
            return

        # Use first stage alphas and entropy
        if len(alphas) > 0:
            alpha_flat = alphas[0].squeeze(-1).squeeze(0).cpu().numpy()  # (N,)
            ent_flat = entropy_maps.squeeze(0).cpu().numpy()  # (N,)

            all_entropy.extend(ent_flat.tolist())
            all_alpha.extend(alpha_flat.tolist())

    if not all_entropy:
        print("[WARNING] No data collected for scatter plot.")
        return

    all_entropy = np.array(all_entropy)
    all_alpha = np.array(all_alpha)

    # Compute correlation
    corr = np.corrcoef(all_entropy, all_alpha)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_entropy, all_alpha, alpha=0.3, s=5, c="steelblue")
    ax.set_xlabel("Patch Pixel-Entropy (H)", fontsize=12)
    ax.set_ylabel("Routing Coefficient (α)", fontsize=12)
    ax.set_title(f"Entropy vs Routing α — "
                 f"Pearson ρ = {corr:.3f}", fontsize=14)
    ax.set_ylim(-0.05, 1.05)

    # Trend line
    z = np.polyfit(all_entropy, all_alpha, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_entropy.min(), all_entropy.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Linear fit (ρ={corr:.3f})")
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "entropy_vs_alpha.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path} (ρ = {corr:.3f})")


# ---------------------------------------------------------------------------
# Training curves from CSV log
# ---------------------------------------------------------------------------

def plot_training_curves(csv_path: str, save_dir: str):
    """Plot loss, accuracy, and mean α curves from the training CSV log."""
    if not HAS_MPL:
        print("[WARNING] matplotlib required for training curves.")
        return

    os.makedirs(save_dir, exist_ok=True)

    steps, loss, top1, top5, alpha, lr = [], [], [], [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))
            top1.append(float(row["top1_acc"]))
            top5.append(float(row["top5_acc"]))
            alpha.append(float(row["mean_alpha"]))
            lr.append(float(row["lr"]))

    if not steps:
        print("[WARNING] No data in CSV log.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curve
    axes[0, 0].plot(steps, loss, "b-", linewidth=0.5)
    axes[0, 0].set_title("Training Loss", fontsize=12)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(steps, top1, "g-", linewidth=0.5, label="Top-1")
    axes[0, 1].plot(steps, top5, "b-", linewidth=0.5, label="Top-5")
    axes[0, 1].set_title("Accuracy", fontsize=12)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Mean α over training
    axes[1, 0].plot(steps, alpha, "r-", linewidth=0.5)
    axes[1, 0].set_title("Mean Routing α", fontsize=12)
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("α")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(steps, lr, "purple", linewidth=0.5)
    axes[1, 1].set_title("Learning Rate", fontsize=12)
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("ESH-Vision Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate ESH-Vision paper assets")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained checkpoint (.pt)")
    p.add_argument("--log_csv", type=str, default=None,
                   help="Path to train_log.csv")
    p.add_argument("--hf_dataset", type=str, default="clane9/imagenet-100")
    p.add_argument("--hf_split", type=str, default="train")
    p.add_argument("--num_images", type=int, default=16,
                   help="Number of images for heatmaps")
    p.add_argument("--num_scatter", type=int, default=50,
                   help="Number of images for scatter plot")
    p.add_argument("--output_dir", type=str, default="./paper_assets")
    p.add_argument("--curves_only", action="store_true",
                   help="Only plot training curves (no model needed)")
    p.add_argument("--img_size", type=int, default=224)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Training curves (always available if CSV exists) -------------------
    log_csv = args.log_csv
    if log_csv is None and args.checkpoint:
        # Try to find CSV next to checkpoint
        ckpt_dir = os.path.dirname(args.checkpoint)
        candidate = os.path.join(ckpt_dir, "train_log.csv")
        if os.path.isfile(candidate):
            log_csv = candidate

    if log_csv and os.path.isfile(log_csv):
        print("\n[1/3] Generating training curves...")
        plot_training_curves(log_csv, args.output_dir)
    else:
        print("[1/3] No CSV log found — skipping training curves.")

    if args.curves_only:
        print("\nDone (curves only mode).")
        return

    # --- Load model & data for heatmaps + scatter --------------------------
    if not args.checkpoint:
        sys.exit("Provide --checkpoint for heatmaps and scatter plots.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config_dict = ckpt["config"]
    config = ESHVisionConfig(**config_dict)
    model = ESHVisionBackbone(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded: {model.get_num_params():,} params")

    # Load sample images
    if not HAS_TV:
        sys.exit("torchvision required for loading images.")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if HAS_HF:
        print(f"Loading images from {args.hf_dataset}...")
        raw_ds = hf_load_dataset(args.hf_dataset, trust_remote_code=True)
        split = args.hf_split if args.hf_split in raw_ds else list(raw_ds.keys())[0]
        n_total = max(args.num_images, args.num_scatter)

        images = []
        for i in range(min(n_total, len(raw_ds[split]))):
            img = raw_ds[split][i]["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(transform(img))

        images_tensor = torch.stack(images)
    else:
        sys.exit("HuggingFace datasets required for loading sample images.")

    # --- Saliency heatmaps --------------------------------------------------
    print(f"\n[2/3] Generating saliency heatmaps ({args.num_images} images)...")
    generate_saliency_heatmaps(
        model, images_tensor, os.path.join(args.output_dir, "heatmaps"),
        device, num_images=args.num_images,
    )

    # --- Entropy vs α scatter -----------------------------------------------
    print(f"\n[3/3] Generating entropy vs α scatter ({args.num_scatter} images)...")
    generate_entropy_scatter(
        model, images_tensor, args.output_dir,
        device, num_images=args.num_scatter,
    )

    print(f"\nAll assets saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
