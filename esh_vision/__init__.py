"""
ESH-Vision: Spatial Entropy-Steered Hybridization
==================================================

A novel vision backbone that dynamically routes image patches between
a Vision State Space Model (VSSM) path and a Multi-Head Self-Attention
(MHSA) path based on learned spatial complexity.

Key idea — "Elastic Intelligence":
  • Simple patches  → SSM (cheap, linear-time)
  • Complex patches → Attention + Adaptive Pondering (powerful, quadratic)
"""

from esh_vision.layers import (
    PatchEmbedding,
    SpatialSoftEntropyRouter,
    BidirectionalVSSM,
    RelativeMHSA,
    HybridVisionBlock,
)
from esh_vision.model import ESHVisionConfig, ESHVisionBackbone

__version__ = "0.1.0"

__all__ = [
    "PatchEmbedding",
    "SpatialSoftEntropyRouter",
    "BidirectionalVSSM",
    "RelativeMHSA",
    "HybridVisionBlock",
    "ESHVisionConfig",
    "ESHVisionBackbone",
]
