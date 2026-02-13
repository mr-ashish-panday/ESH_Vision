"""
ESH-Vision Backbone Model
=========================

Hierarchical vision backbone with configurable depth, dimensions, and
entropy-steered hybrid blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from esh_vision.layers import HybridVisionBlock, PatchEmbedding


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ESHVisionConfig:
    """All hyperparameters for an ESH-Vision backbone.

    Defaults yield a *Large* variant (~500 M parameters) suitable for
    12 GB VRAM training at batch-size 4 with gradient accumulation
    and gradient checkpointing.
    """

    # Input
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3

    # Architecture — 3-stage hierarchy (dims: 384 → 768 → 1536)
    embed_dim: int = 384
    depths: List[int] = field(default_factory=lambda: [4, 8, 10])
    num_heads: List[int] = field(default_factory=lambda: [6, 12, 12])
    dim_scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # SSM
    ssm_d_state: int = 16

    # ACT
    act_threshold: float = 0.7
    max_ponder: int = 2

    # Regularisation
    drop_rate: float = 0.1
    mlp_ratio: float = 4.0

    # Hardware
    use_checkpoint: bool = True

    # Number of classes (0 = backbone-only, no head)
    num_classes: int = 0


# ---------------------------------------------------------------------------
# Downsampling between stages
# ---------------------------------------------------------------------------

class PatchMerging(nn.Module):
    """Spatial downsampling: 2×2 patch groups → concatenate → linear project.

    Reduces spatial resolution by 2× and doubles channel dimension.
    """

    def __init__(self, dim: int, grid_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, H*W, D) where H = W = grid_size.

        Returns
        -------
        out : (B, ceil(H/2)*ceil(W/2), 2*D)
        """
        B, N, D = x.shape
        H = W = self.grid_size
        assert N == H * W, f"Expected N={H*W}, got {N}"

        x = x.view(B, H, W, D)

        # Pad to even dimensions if needed
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
            H, W = H + pad_h, W + pad_w

        # Gather 2×2 patches
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, D)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4D)

        x = x.view(B, -1, 4 * D)
        x = self.norm(x)
        x = self.reduction(x)  # (B, (H/2)*(W/2), 2D)

        return x


# ---------------------------------------------------------------------------
# ESH-Vision Backbone
# ---------------------------------------------------------------------------

class ESHVisionBackbone(nn.Module):
    """Hierarchical ESH-Vision backbone.

    Architecture::

        Image → PatchEmbedding → [Stage₁ → Downsample → Stage₂ → ... ] → Norm → Pool

    Each stage is a stack of ``HybridVisionBlock`` modules. Downsampling
    (``PatchMerging``) halves spatial resolution and doubles channel dim.

    Returns a dict with:
        - ``features``    : (B, D_final) pooled feature vector
        - ``alphas``      : list of (B, N_stage, 1) routing weights per block
        - ``ponder_cost`` : scalar accumulated ponder cost
    """

    def __init__(self, config: ESHVisionConfig):
        super().__init__()
        self.config = config

        # --- Patch embedding ------------------------------------------------
        self.patch_embed = PatchEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
        )
        grid_size = config.img_size // config.patch_size  # e.g. 14

        # --- Build stages ---------------------------------------------------
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur_dim = config.embed_dim
        cur_grid = grid_size

        for stage_idx, depth in enumerate(config.depths):
            dim = int(config.embed_dim * config.dim_scale[stage_idx])
            if stage_idx > 0:
                dim = cur_dim  # already scaled by PatchMerging

            # Projection if dim changes from patch embed
            blocks = nn.ModuleList()
            for _ in range(depth):
                blocks.append(
                    HybridVisionBlock(
                        embed_dim=cur_dim,
                        num_heads=config.num_heads[stage_idx],
                        grid_size=cur_grid,
                        ssm_d_state=config.ssm_d_state,
                        mlp_ratio=config.mlp_ratio,
                        act_threshold=config.act_threshold,
                        max_ponder=config.max_ponder,
                        drop_rate=config.drop_rate,
                    )
                )
            self.stages.append(blocks)

            # Downsample between stages (except last)
            if stage_idx < len(config.depths) - 1:
                ds = PatchMerging(cur_dim, cur_grid)
                self.downsamples.append(ds)
                cur_dim = cur_dim * 2
                cur_grid = (cur_grid + 1) // 2  # ceiling division (matches PatchMerging padding)

        self.final_dim = cur_dim
        self.norm = nn.LayerNorm(cur_dim)

        # Optional classification head
        self.head = (
            nn.Linear(cur_dim, config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        """Sensible weight initialisation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, images: torch.Tensor
    ) -> Dict[str, object]:
        """
        Parameters
        ----------
        images : (B, C, H, W) float tensor, pixel values in [0, 1].

        Returns
        -------
        dict with keys:
            features    : (B, D_final) or (B, num_classes) if head exists
            alphas      : list[(B, N_i, 1)] per block
            ponder_cost : scalar tensor
        """
        # Patch embedding + entropy
        x, entropy = self.patch_embed(images)  # x: (B, N, D), entropy: (B, N)

        all_alphas: List[torch.Tensor] = []
        total_ponder = torch.tensor(0.0, device=x.device)

        for stage_idx, blocks in enumerate(self.stages):
            for block in blocks:
                if self.config.use_checkpoint and self.training:
                    # Gradient checkpointing — saves VRAM at compute cost
                    def create_forward(blk):
                        def custom_fwd(x_in, ent_in):
                            return blk(x_in, ent_in)
                        return custom_fwd

                    x, alpha, ponder = cp.checkpoint(
                        create_forward(block), x, entropy,
                        use_reentrant=False,
                    )
                else:
                    x, alpha, ponder = block(x, entropy)

                all_alphas.append(alpha)
                total_ponder = total_ponder + ponder

            # Downsample (except after last stage)
            if stage_idx < len(self.stages) - 1:
                x = self.downsamples[stage_idx](x)
                # Entropy must match new spatial size of x
                new_n = x.shape[1]
                B_ent = entropy.shape[0]
                cur_grid = int(math.isqrt(entropy.shape[1]))
                ent_2d = entropy.view(B_ent, cur_grid, cur_grid)
                # Pad to even before pooling (matches PatchMerging)
                pad_h = cur_grid % 2
                pad_w = cur_grid % 2
                if pad_h or pad_w:
                    ent_2d = F.pad(ent_2d.unsqueeze(1), (0, pad_w, 0, pad_h)).squeeze(1)
                else:
                    ent_2d = ent_2d.unsqueeze(1).squeeze(1)
                ent_2d = F.avg_pool2d(
                    ent_2d.unsqueeze(1).float(), kernel_size=2, stride=2
                ).squeeze(1)
                entropy = ent_2d.view(B_ent, -1)
                assert entropy.shape[1] == new_n, f"Entropy {entropy.shape[1]} != x {new_n}"

        # Final norm + global average pool
        x = self.norm(x)                    # (B, N_last, D_final)
        features = x.mean(dim=1)            # (B, D_final)

        # Classification head (identity if num_classes == 0)
        logits = self.head(features)

        return {
            "features": logits,
            "alphas": all_alphas,
            "ponder_cost": total_ponder,
        }

    def get_num_params(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
