# ESH-Vision: Spatial Entropy-Steered Hybridization

A novel vision backbone that **dynamically routes image patches** between a Vision State Space Model (VSSM) path and a Multi-Head Self-Attention (MHSA) path based on learned spatial complexity.

> **Core Hypothesis — "Elastic Intelligence":**  Simple patches → SSM (cheap, linear-time). Complex patches → Attention + Adaptive Pondering (powerful, quadratic).

## Architecture

```
Image → PatchEmbedding (+ pixel-entropy) → [HybridVisionBlock × N] → Features
                                                    │
                                    ┌───────────────┼───────────────┐
                                    │               │               │
                              SpatialRouter    BidirVSSM        RelMHSA
                                 (α_i)            │               │
                                    │        (1-α)·VSSM      α·Attention
                                    │               │               │
                                    └───────────────┴───────┬───────┘
                                                            │
                                                   Weighted Merge + FFN
                                                            │
                                                  ACT Ponder (if α > θ)
```

### Key Components

| Module | Description |
|---|---|
| `SpatialSoftEntropyRouter` | MLP router with pixel-entropy prior → α ∈ [0,1] per patch |
| `BidirectionalVSSM` | Pure-PyTorch Mamba-style S4 selective scan (forward + reverse) |
| `RelativeMHSA` | Multi-Head Self-Attention with Swin-style relative position bias |
| `HybridVisionBlock` | α-weighted routing + Adaptive Computation Time ponder loop |
| `ESHVisionBackbone` | Hierarchical multi-stage backbone with patch merging |

## Project Structure

```
esh_vision/
├── esh_vision/
│   ├── __init__.py          # Package exports
│   ├── layers.py            # Router, VSSM, MHSA, HybridBlock
│   └── model.py             # Full backbone + config
├── train_vision.py          # Training with custom losses & AMP
├── visualize_saliency.py    # α-value heatmap generation
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Smoke Test (no data needed)

```python
import torch
from esh_vision.model import ESHVisionBackbone, ESHVisionConfig

config = ESHVisionConfig(num_classes=1000)
model = ESHVisionBackbone(config)
x = torch.randn(2, 3, 224, 224)
out = model(x)

print(out["features"].shape)   # (2, 1000)
print(out["ponder_cost"])       # scalar
print(len(out["alphas"]))       # 24 (one per block)
```

### Training

```bash
# Smoke test with random data
python train_vision.py --smoke_test --epochs 2 --batch_size 4

# Full ImageNet training
python train_vision.py \
    --data_dir /path/to/imagenet \
    --num_classes 1000 \
    --epochs 90 \
    --batch_size 32 \
    --lr 1e-3
```

### Saliency Visualisation

```bash
# Demo with synthetic image
python visualize_saliency.py --demo

# With a real image + checkpoint
python visualize_saliency.py \
    --checkpoint checkpoints/esh_vision_ep90.pt \
    --image path/to/image.jpg
```

## Training Features

- **Variance-Incentive Loss**: $-\lambda_{var} \cdot \text{Var}(\alpha)$ encourages bimodal routing (decisive SSM vs. Attention choices)
- **Ponder Cost Warmup**: λ starts at 0, linearly ramps after 15k steps
- **AMP**: Automatic Mixed Precision via `torch.cuda.amp`
- **8-bit Optimiser**: `bitsandbytes` AdamW8bit for 12 GB VRAM constraint
- **Gradient Checkpointing**: Block-level checkpointing enabled by default

## Citation

```bibtex
@article{eshvision2026,
  title={ESH-Vision: Spatial Entropy-Steered Hybridization for Efficient Vision Backbones},
  year={2026}
}
```

## License

MIT
