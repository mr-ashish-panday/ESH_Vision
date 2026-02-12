#!/usr/bin/env python3
"""
ESH-Vision Saliency Visualisation
==================================

Generates per-layer α-value heatmaps overlaid on input images, revealing
which patches the model routes to Attention (α→1, red/hot) vs. VSSM
(α→0, blue/cool).

Usage
-----
    python visualize_saliency.py \\
        --checkpoint checkpoints/esh_vision_ep90.pt \\
        --image path/to/image.jpg \\
        --output_dir saliency_maps/

For a quick demo without a checkpoint (random model)::

    python visualize_saliency.py --demo
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from esh_vision.model import ESHVisionBackbone, ESHVisionConfig

try:
    from PIL import Image
    from torchvision import transforms

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Core visualisation
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str | None,
    config: ESHVisionConfig | None = None,
    device: torch.device = torch.device("cpu"),
) -> ESHVisionBackbone:
    """Load an ESH-Vision model from a checkpoint or create a fresh one."""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_config = ckpt.get("config", config or ESHVisionConfig())
        model = ESHVisionBackbone(saved_config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Saliency] Loaded checkpoint: {checkpoint_path}")
    else:
        cfg = config or ESHVisionConfig()
        model = ESHVisionBackbone(cfg).to(device)
        print("[Saliency] Using randomly initialised model (no checkpoint)")
    model.eval()
    return model


def preprocess_image(
    image_path: str, img_size: int = 224
) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess an image for inference.

    Returns
    -------
    tensor   : (1, 3, H, W) normalised tensor.
    original : (H, W, 3) numpy array for overlay.
    """
    if not HAS_PIL:
        raise ImportError("Pillow + torchvision required for image loading")

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)

    # Original image for overlay
    original = np.array(img.resize((img_size, img_size))) / 255.0

    return tensor, original


def create_demo_image(img_size: int = 224) -> tuple[torch.Tensor, np.ndarray]:
    """Create a synthetic test image with varying complexity regions."""
    # Left half: smooth gradient (low entropy → should route to VSSM)
    # Right half: checkerboard noise (high entropy → should route to Attention)
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)

    # Smooth gradient (left)
    for i in range(img_size):
        img[:, i, 0] = i / img_size * 0.6 + 0.2
        img[:, i, 1] = 0.3
        img[:, i, 2] = 1.0 - i / img_size * 0.4

    # Noisy region (right half)
    half = img_size // 2
    noise = np.random.RandomState(42).rand(img_size, half, 3).astype(np.float32)
    img[:, half:, :] = noise * 0.8 + 0.1

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor, img


def extract_alphas(
    model: ESHVisionBackbone,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> list[np.ndarray]:
    """Run inference and extract per-block α maps.

    Returns
    -------
    alpha_maps : list of (gH, gW) numpy arrays, one per block.
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        out = model(image_tensor)

    alpha_maps = []
    for alpha in out["alphas"]:
        # alpha: (B, N, 1) → (N,)
        a = alpha[0, :, 0].cpu().numpy()
        grid_side = int(math.isqrt(len(a)))
        if grid_side * grid_side == len(a):
            a_2d = a.reshape(grid_side, grid_side)
        else:
            # Non-square case — reshape to closest rectangle
            a_2d = a.reshape(1, -1)
        alpha_maps.append(a_2d)

    return alpha_maps


def plot_saliency(
    original: np.ndarray,
    alpha_maps: list[np.ndarray],
    output_dir: str,
    img_name: str = "input",
):
    """Create and save α-value heatmap overlays.

    Generates:
      - Per-layer heatmaps
      - A composite summary (first, middle, last layers)
    """
    os.makedirs(output_dir, exist_ok=True)
    num_layers = len(alpha_maps)

    # --- Per-layer heatmaps -------------------------------------------------
    for idx, a_map in enumerate(alpha_maps):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(original)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Alpha heatmap
        h_img = axes[1].imshow(a_map, cmap="RdYlBu_r", vmin=0, vmax=1,
                                interpolation="nearest")
        axes[1].set_title(f"α Map — Block {idx+1}/{num_layers}")
        axes[1].axis("off")
        plt.colorbar(h_img, ax=axes[1], fraction=0.046)

        # Overlay
        axes[2].imshow(original)
        # Upsample alpha map to image size for overlay
        a_up = np.kron(
            a_map,
            np.ones((
                original.shape[0] // a_map.shape[0],
                original.shape[1] // a_map.shape[1],
            ))
        )
        # Handle edge case where kron doesn't match exactly
        a_up = a_up[:original.shape[0], :original.shape[1]]
        axes[2].imshow(a_up, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title(f"Overlay — Block {idx+1}")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = Path(output_dir) / f"{img_name}_block_{idx+1:02d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    # --- Composite summary --------------------------------------------------
    key_layers = [0, num_layers // 2, num_layers - 1]
    key_layers = list(dict.fromkeys(key_layers))  # deduplicate

    fig, axes = plt.subplots(1, len(key_layers) + 1, figsize=(5 * (len(key_layers) + 1), 5))

    axes[0].imshow(original)
    axes[0].set_title("Input", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    for i, layer_idx in enumerate(key_layers):
        a_map = alpha_maps[layer_idx]
        a_up = np.kron(
            a_map,
            np.ones((
                original.shape[0] // a_map.shape[0],
                original.shape[1] // a_map.shape[1],
            ))
        )
        a_up = a_up[:original.shape[0], :original.shape[1]]

        axes[i + 1].imshow(original)
        im = axes[i + 1].imshow(a_up, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        label = {0: "Early", num_layers // 2: "Middle", num_layers - 1: "Late"}
        axes[i + 1].set_title(
            f"{label.get(layer_idx, '')} Block {layer_idx+1}",
            fontsize=14, fontweight="bold",
        )
        axes[i + 1].axis("off")

    plt.suptitle(
        "ESH-Vision Routing Saliency — α Values\n"
        "Hot (α→1) = Attention  |  Cool (α→0) = VSSM",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    composite_path = Path(output_dir) / f"{img_name}_composite.png"
    plt.savefig(composite_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved composite: {composite_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="ESH-Vision Saliency Visualisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to model checkpoint (.pt)")
    p.add_argument("--image", type=str, default=None,
                    help="Path to input image")
    p.add_argument("--output_dir", type=str, default="./saliency_maps",
                    help="Directory to save heatmaps")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--demo", action="store_true",
                    help="Run with synthetic test image (no checkpoint needed)")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint, device=device)

    # Load or create image
    if args.demo:
        print("[Saliency] Demo mode — generating synthetic test image")
        tensor, original = create_demo_image(args.img_size)
        img_name = "demo"
    elif args.image:
        tensor, original = preprocess_image(args.image, args.img_size)
        img_name = Path(args.image).stem
    else:
        print("Error: Provide --image or --demo")
        return

    # Extract alpha maps
    print(f"[Saliency] Running inference ({len(model.config.depths)} stages, "
          f"{sum(model.config.depths)} blocks)...")
    alpha_maps = extract_alphas(model, tensor, device)

    # Plot
    print(f"[Saliency] Generating {len(alpha_maps)} heatmaps...")
    plot_saliency(original, alpha_maps, args.output_dir, img_name=img_name)

    print(f"\n[Saliency] All done! Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
