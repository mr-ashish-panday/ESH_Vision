
import argparse
import time
import torch
import torch.cuda
from esh_vision.model import ESHVisionBackbone, ESHVisionConfig

def benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        
    cfg = ESHVisionConfig(
        embed_dim=256,
        depths=[3, 6, 6],
        num_heads=[4, 8, 8],
        num_classes=100,
        use_checkpoint=True
    )
    model = ESHVisionBackbone(cfg).to(device)
    model.train()
    
    B = args.batch_size
    print(f"\nBenchmarking Model (Batch={B}, AMP={args.amp})...")
    
    x = torch.randn(B, 3, 224, 224, device=device)
    target = torch.randint(0, 100, (B,), device=device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    print("Warmup...")
    for _ in range(5):
        with torch.cuda.amp.autocast(enabled=args.amp):
            out = model(x)
            loss = out["features"].sum()
        scaler.scale(loss).backward()
        model.zero_grad()
        
    torch.cuda.synchronize()
    start = time.time()
    steps = 20
    
    print(f"Running {steps} steps...")
    for i in range(steps):
        with torch.cuda.amp.autocast(enabled=args.amp):
            out = model(x)
            loss = out["features"].sum()
        scaler.scale(loss).backward()
        model.zero_grad()
        
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Time per step: {elapsed/steps:.3f}s")
    print(f"Throughput: {B * steps / elapsed:.1f} img/s")
    
if __name__ == "__main__":
    benchmark()
