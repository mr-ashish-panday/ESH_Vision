
import time
import torch
import torch.cuda
from esh_vision.model import ESHVisionBackbone, ESHVisionConfig

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # Config matching the training run
    cfg = ESHVisionConfig(
        embed_dim=256,
        depths=[3, 6, 6],
        num_heads=[4, 8, 8],
        num_classes=100,
        use_checkpoint=True
    )
    model = ESHVisionBackbone(cfg).to(device)
    model.train()
    
    # Dummy batch matching training command (Batch 48)
    B = 48
    x = torch.randn(B, 3, 224, 224, device=device)
    target = torch.randint(0, 100, (B,), device=device)
    
    print(f"\nBenchmarking Model (Batch={B})...")
    print("Warmup...")
    for _ in range(5):
        out = model(x)
        features = out["features"]
        loss = features.sum()
        loss.backward()
        model.zero_grad()
        
    torch.cuda.synchronize()
    start = time.time()
    steps = 20
    
    print(f"Running {steps} steps...")
    for i in range(steps):
        out = model(x)
        features = out["features"]
        loss = features.sum()
        loss.backward()
        model.zero_grad()
        
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Time per step: {elapsed/steps:.3f}s")
    print(f"Throughput: {B * steps / elapsed:.1f} img/s")
    
if __name__ == "__main__":
    benchmark()
