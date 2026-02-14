"""
ESH-Vision Core Layers
======================

Modules
-------
PatchEmbedding           – Conv2d patch tokenizer with per-patch entropy
SpatialSoftEntropyRouter – MLP router producing per-patch α ∈ [0, 1]
BidirectionalVSSM        – Pure-PyTorch Mamba-style bidirectional S4 scan
RelativeMHSA             – Multi-Head Self-Attention with relative pos bias
HybridVisionBlock        – Entropy-steered routing with ACT ponder loop
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# 1. Patch Embedding with Pixel-Entropy Side-Channel
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Tokenise an image into patch embeddings and compute per-patch entropy.

    Parameters
    ----------
    img_size   : int   – expected spatial resolution (square).
    patch_size : int   – side length of each (non-overlapping) patch.
    in_chans   : int   – number of input channels (3 for RGB).
    embed_dim  : int   – output embedding dimensionality D.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # Patch projection (non-overlapping conv)
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _patch_entropy(images: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Compute Shannon entropy of raw pixel intensities per patch.

        Parameters
        ----------
        images     : (B, C, H, W) float tensor in [0, 1].
        patch_size : int

        Returns
        -------
        entropy : (B, N) tensor, one scalar per patch.
        """
        B, C, H, W = images.shape
        gH, gW = H // patch_size, W // patch_size

        # Convert to grayscale for entropy computation
        gray = images.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Unfold into patches: (B, 1, gH, patch_size, gW, patch_size)
        patches = gray.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, gH * gW, -1)  # (B, N, patch_size^2)

        # Quantise to 32 bins for a lightweight histogram
        num_bins = 32
        patches_q = (patches * (num_bins - 1)).clamp(0, num_bins - 1).long()

        # One-hot → histogram → probabilities
        one_hot = F.one_hot(patches_q, num_classes=num_bins).float()  # (B, N, P, bins)
        hist = one_hot.sum(dim=2)  # (B, N, bins)
        prob = hist / hist.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Shannon entropy: H = -Σ p log₂ p
        log_prob = torch.log2(prob + 1e-10)
        entropy = -(prob * log_prob).sum(dim=-1)  # (B, N)

        # Normalise to [0, 1] by dividing by max possible entropy
        max_entropy = math.log2(num_bins)
        entropy = entropy / max_entropy

        return entropy

    # ---- forward -----------------------------------------------------------

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        images : (B, C, H, W) float tensor, pixel values in [0, 1].

        Returns
        -------
        x       : (B, N, D)  – patch embeddings.
        entropy : (B, N)     – per-patch pixel entropy (normalised [0, 1]).
        """
        # Pixel entropy (computed before projection, on raw pixels)
        entropy = self._patch_entropy(images, self.patch_size)  # (B, N)

        # Patch projection + flatten
        x = self.proj(images)                       # (B, D, gH, gW)
        x = rearrange(x, "b d h w -> b (h w) d")   # (B, N, D)
        x = self.norm(x)

        return x, entropy


# ---------------------------------------------------------------------------
# 2. Spatial Soft-Entropy Router
# ---------------------------------------------------------------------------

class SpatialSoftEntropyRouter(nn.Module):
    """Lightweight MLP that predicts a routing scalar α_i ∈ [0, 1] per patch.

    The router receives both the D-dim patch embedding *and* the scalar
    pixel-entropy as a "+1" input channel — the "entropy prior".

    α → 1 means "use Attention", α → 0 means "use VSSM".

    Parameters
    ----------
    embed_dim  : int – patch embedding dimension D.
    hidden_dim : int – hidden layer width (default: D // 4).
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim // 4, 32)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, entropy: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (B, N, D) patch embeddings.
        entropy : (B, N)    per-patch pixel entropy.

        Returns
        -------
        alpha : (B, N, 1) routing weights in [0, 1].
        """
        # Concatenate entropy as an extra feature
        ent = entropy.unsqueeze(-1)             # (B, N, 1)
        inp = torch.cat([x, ent], dim=-1)       # (B, N, D+1)
        alpha = torch.sigmoid(self.net(inp))    # (B, N, 1)
        return alpha


# ---------------------------------------------------------------------------
# 3. Bidirectional Vision State Space Model (VSSM)
# ---------------------------------------------------------------------------

class BidirectionalVSSM(nn.Module):
    """Pure-PyTorch quad-directional S4/Mamba-style selective scan for 2D.

    Scans the flattened patch sequence in 4 directions to capture full 2D
    spatial context:
      1. Row-major forward   (top-left → bottom-right)
      2. Row-major reverse   (bottom-right → top-left)
      3. Column-major forward (top-left → bottom-right, column-first)
      4. Column-major reverse (bottom-right → top-left, column-first)

    All 4 directions share SSM parameters — diversity comes from scan
    ordering alone, so there is zero parameter increase over 2-way.

    Parameters
    ----------
    d_model   : int – input/output dimension D.
    d_state   : int – SSM latent state dimension N (default 16).
    d_inner   : int – expanded inner dimension E (default 2*D).
    dt_rank   : int – rank of the Δt projection (default D // 16).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_inner: Optional[int] = None,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner or d_model * 2
        self.dt_rank = dt_rank or max(d_model // 16, 1)

        # Input projection: project to inner dim
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM parameters (shared across all 4 scan directions)
        # A is initialised as the HiPPO-LegS matrix diagonal (log-space)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0).expand(self.d_inner, -1).clone()
        )  # (E, N)

        # B, C projections from input
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # Δt projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise dt bias so that softplus(dt_bias) ≈ [0.001 .. 0.1]
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            )
            inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_softplus)

        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _ssm_scan(
        self,
        x: torch.Tensor,  # (B, L, E)
        reverse: bool = False,
    ) -> torch.Tensor:
        """Run a selective scan (linear recurrence) over the sequence.

        Parameters
        ----------
        x       : (B, L, E) input tensor.
        reverse : bool – if True, scan in reverse order.

        Returns
        -------
        y : (B, L, E) output tensor.
        """
        B, L, E = x.shape
        N = self.d_state

        # Compute Δ, B, C from input
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, N, N], dim=-1
        )

        # Δt: project and apply softplus
        dt = F.softplus(self.dt_proj(dt))  # (B, L, E)

        # A (discretised): A_bar = exp(A * dt)
        A = -torch.exp(self.A_log)  # (E, N)
        # Broadcasting: dt is (B, L, E), A is (E, N) → dt_A is (B, L, E, N)
        dt_A = torch.einsum("ble,en->blen", dt, A)
        A_bar = torch.exp(dt_A)  # (B, L, E, N)

        # B discretised: B_bar = dt * B
        dt_B = torch.einsum("ble,bln->blen", dt, B_param)  # (B, L, E, N)

        # Iterate (sequential recurrence — checkpoint-safe)
        indices = range(L - 1, -1, -1) if reverse else range(L)

        h = torch.zeros(B, E, N, device=x.device, dtype=x.dtype)  # (B, E, N)
        ys = []

        for i in indices:
            # h = A_bar * h + B_bar * x
            h = A_bar[:, i] * h + dt_B[:, i] * x[:, i].unsqueeze(-1)  # (B, E, N)
            # y = C * h
            y_i = torch.einsum("ben,bn->be", h, C_param[:, i])  # (B, E)
            ys.append(y_i)

        if reverse:
            ys = ys[::-1]

        y = torch.stack(ys, dim=1)  # (B, L, E)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N_patches, D) – patch embeddings.

        Returns
        -------
        out : (B, N_patches, D) – processed embeddings.
        """
        B, L, D = x.shape
        H = W = int(math.sqrt(L))
        residual = x

        # Input projection → x_inner, z (gating)
        xz = self.in_proj(x)  # (B, L, 2*E)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, E)

        # --- Row-major scans (original bidirectional) -----------------------
        y_row_fwd = self._ssm_scan(x_inner, reverse=False)
        y_row_rev = self._ssm_scan(x_inner, reverse=True)

        # --- Column-major scans (transpose H↔W, scan, transpose back) ------
        E = x_inner.shape[-1]
        x_col = x_inner.view(B, H, W, E).transpose(1, 2).contiguous().view(B, L, E)
        y_col_fwd_t = self._ssm_scan(x_col, reverse=False)
        y_col_rev_t = self._ssm_scan(x_col, reverse=True)
        # Transpose back to row-major order
        y_col_fwd = y_col_fwd_t.view(B, W, H, E).transpose(1, 2).contiguous().view(B, L, E)
        y_col_rev = y_col_rev_t.view(B, W, H, E).transpose(1, 2).contiguous().view(B, L, E)

        # Merge all 4 directions (average)
        y = (y_row_fwd + y_row_rev + y_col_fwd + y_col_rev) * 0.25

        # Gated residual + D skip
        y = y * F.silu(z) + x_inner * self.D  # (B, L, E)

        # Output projection
        out = self.out_proj(y)  # (B, L, D)
        out = self.norm(out + residual)

        return out


# ---------------------------------------------------------------------------
# 4. Relative-Position Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class RelativeMHSA(nn.Module):
    """Standard MHSA with learnable relative position bias (Swin-style).

    Parameters
    ----------
    embed_dim   : int – model dimension D.
    num_heads   : int – number of attention heads.
    grid_size   : int – spatial grid side (e.g. 14 for 224/16).
    attn_drop   : float – dropout on attention weights.
    proj_drop   : float – dropout on output projection.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        grid_size: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(embed_dim)

        # Relative position bias table
        # (2*gH-1) * (2*gW-1) entries, one per head
        num_rel = (2 * grid_size - 1) * (2 * grid_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_rel, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index (fixed)
        coords_h = torch.arange(grid_size)
        coords_w = torch.arange(grid_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, gH, gW)
        coords_flat = coords.view(2, -1)  # (2, N)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel_coords[0] += grid_size - 1
        rel_coords[1] += grid_size - 1
        rel_coords[0] *= 2 * grid_size - 1
        rel_index = rel_coords.sum(0)  # (N, N)
        self.register_buffer("relative_position_index", rel_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, D) – patch embeddings.

        Returns
        -------
        out : (B, N, D) – attended embeddings (with residual + norm).
        """
        residual = x
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Add relative position bias
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # (N, N, num_heads)
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)
        attn = attn + rel_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj_drop(self.proj(out))
        out = self.norm(out + residual)

        return out


# ---------------------------------------------------------------------------
# 5. Hybrid Vision Block with ACT (Adaptive Computation Time)
# ---------------------------------------------------------------------------

class HybridVisionBlock(nn.Module):
    """Core ESH-Vision routing block.

    Routing formula per patch *i*::

        output_i = α_i · Attention(X) + (1 - α_i) · VSSM(X)

    ACT (Adaptive Computation Time):
        Patches with α > ``act_threshold`` are recursively re-processed
        through this block up to ``max_ponder`` additional times. A halting
        probability accumulates, and excess computation is penalised via a
        ponder cost added to the training loss.

    Parameters
    ----------
    embed_dim      : int   – model dimension D.
    num_heads      : int   – heads for MHSA.
    grid_size      : int   – spatial grid side.
    ssm_d_state    : int   – SSM latent state dim N.
    mlp_ratio      : float – FFN hidden dim = embed_dim * mlp_ratio.
    act_threshold  : float – α threshold for triggering pondering.
    max_ponder     : int   – max extra passes for high-entropy patches.
    drop_rate      : float – dropout rate.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        grid_size: int,
        ssm_d_state: int = 16,
        mlp_ratio: float = 4.0,
        act_threshold: float = 0.7,
        max_ponder: int = 2,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.act_threshold = act_threshold
        self.max_ponder = max_ponder

        # Sub-modules
        self.router = SpatialSoftEntropyRouter(embed_dim)
        self.attn = RelativeMHSA(
            embed_dim, num_heads, grid_size,
            attn_drop=drop_rate, proj_drop=drop_rate,
        )
        self.vssm = BidirectionalVSSM(embed_dim, d_state=ssm_d_state)

        # Feed-forward network
        ffn_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(ffn_hidden, embed_dim),
            nn.Dropout(drop_rate),
        )

        self.pre_norm = nn.LayerNorm(embed_dim)

        # Halting probability projection for ACT (maps D → 1 → sigmoid)
        self.halt_proj = nn.Linear(embed_dim, 1)

    def _single_pass(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One routing + FFN pass.

        Returns
        -------
        out   : (B, N, D) processed patches.
        alpha : (B, N, 1) routing weights.
        """
        normed = self.pre_norm(x)

        # Routing decision
        alpha = self.router(normed, entropy)  # (B, N, 1)

        # Both paths process the *full* sequence (required for global context)
        attn_out = self.attn(normed)   # (B, N, D)
        vssm_out = self.vssm(normed)   # (B, N, D)

        # Weighted merge
        merged = alpha * attn_out + (1 - alpha) * vssm_out  # (B, N, D)

        # Residual + FFN
        out = x + merged
        out = out + self.ffn(out)

        return out, alpha

    def forward(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x       : (B, N, D) – input patch embeddings.
        entropy : (B, N)    – per-patch pixel entropy.

        Returns
        -------
        output      : (B, N, D) – output embeddings.
        alpha       : (B, N, 1) – final routing weights.
        ponder_cost : scalar    – ACT ponder cost (for loss).
        """
        B, N, D = x.shape
        device = x.device

        # First pass (mandatory)
        out, alpha = self._single_pass(x, entropy)

        # --- Adaptive Computation Time (ACT) for high-entropy patches ------
        # Identify patches that should ponder (alpha > threshold)
        needs_ponder = (alpha.squeeze(-1) > self.act_threshold)  # (B, N) bool

        if self.training and needs_ponder.any() and self.max_ponder > 0:
            # Accumulated halting probability
            halting_prob = torch.zeros(B, N, device=device)
            remainders = torch.zeros(B, N, device=device)
            n_updates = torch.zeros(B, N, device=device)

            still_pondering = needs_ponder.float()  # (B, N)

            for step in range(self.max_ponder):
                # Compute halting probability for pondering patches
                halt_p = torch.sigmoid(
                    self.halt_proj(out).squeeze(-1)
                )  # (B, N)

                # Only update patches that are still pondering
                halt_p = halt_p * still_pondering

                # Check which patches will halt this step
                new_halted = (halting_prob + halt_p > 1.0) & (still_pondering > 0.5)

                # For newly halted: remainder = 1 - halting_prob
                remainders = torch.where(
                    new_halted, 1.0 - halting_prob, remainders
                )

                # For still going: add halt_p
                halting_prob = torch.where(
                    new_halted,
                    torch.ones_like(halting_prob),
                    halting_prob + halt_p,
                )

                n_updates = n_updates + still_pondering

                # Update still_pondering
                still_pondering = still_pondering * (~new_halted).float()

                if still_pondering.sum() < 0.5:
                    break

                # Re-process pondering patches
                # Mask to only update pondering patches (in-place)
                ponder_mask = still_pondering.unsqueeze(-1)  # (B, N, 1)
                ponder_out, _ = self._single_pass(out, entropy)
                out = out * (1 - ponder_mask) + ponder_out * ponder_mask

            # Ponder cost = mean number of updates for pondered patches
            ponder_cost = (n_updates + remainders).sum() / max(needs_ponder.sum().item(), 1.0)
        else:
            ponder_cost = torch.tensor(0.0, device=device)

        return out, alpha, ponder_cost
