#!/usr/bin/env python3

import torch
import numpy as np
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[0]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ofc_ml.network import HybridFNOKANPredictor

def test_gpu_computation():
    print("="*80)
    print("GPU 性能诊断")
    print("="*80)
    
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA 不可用！请检查:")
        print("  1. 是否安装了支持CUDA的PyTorch版本")
        print("  2. NVIDIA驱动是否正确安装")
        print("  3. 运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return
    
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-processor count: {props.multi_processor_count}")
    
    print(f"\n当前GPU: {torch.cuda.current_device()}")
    print(f"GPU Memory Free: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print("\n" + "="*80)
    print("测试 1: 简单矩阵乘法")
    print("="*80)
    
    size = 4096
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    warmup = 10
    iterations = 100
    
    for _ in range(warmup):
        _ = torch.mm(a, b)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        c = torch.mm(a, b)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time = elapsed / iterations
    flops = 2 * size**3
    gflops = flops / avg_time / 1e9
    
    print(f"矩阵大小: {size}x{size}")
    print(f"平均时间: {avg_time*1000:.2f} ms")
    print(f"性能: {gflops:.2f} GFLOPS")
    
    if gflops < 100:
        print(f"⚠️  性能偏低（正常应该 >100 GFLOPS）")
    else:
        print(f"✅ 性能正常")
    
    print("\n" + "="*80)
    print("测试 2: 模型前向传播")
    print("="*80)
    
    batch_size = 128
    input_dim = 113
    output_dim = 95
    
    model = HybridFNOKANPredictor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[256, 256, 128, 128, 64],
        dropout=0.2,
        use_residual=True,
        n_frequencies=4,
        spectral_freq_ratio=0.5,
        use_spectral_mixing=True,
    ).cuda()
    
    model.eval()
    
    x = torch.randn(batch_size, input_dim, device='cuda')
    mask = torch.randn(batch_size, output_dim, device='cuda')
    
    for _ in range(warmup):
        _ = model(x, mask)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(x, mask)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time = elapsed / iterations
    samples_per_sec = batch_size / avg_time
    
    print(f"Batch size: {batch_size}")
    print(f"平均时间: {avg_time*1000:.2f} ms")
    print(f"Samples/sec: {samples_per_sec:.1f}")
    print(f"GPU Memory Used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    if samples_per_sec < 1000:
        print(f"⚠️  性能偏低（正常应该 >1000 samples/sec）")
    else:
        print(f"✅ 性能正常")
    
    print("\n" + "="*80)
    print("测试 3: 模型训练（前向+反向）")
    print("="*80)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(batch_size, input_dim, device='cuda', requires_grad=True)
    mask = torch.randn(batch_size, output_dim, device='cuda')
    y = torch.randn(batch_size, output_dim, device='cuda')
    
    for _ in range(warmup):
        optimizer.zero_grad()
        output = model(x, mask)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    
    for _ in range(iterations):
        optimizer.zero_grad()
        output = model(x, mask)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time = elapsed / iterations
    samples_per_sec = batch_size / avg_time
    max_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Batch size: {batch_size}")
    print(f"平均时间: {avg_time*1000:.2f} ms")
    print(f"Samples/sec: {samples_per_sec:.1f}")
    print(f"GPU Memory Peak: {max_memory:.2f} GB")
    
    if samples_per_sec < 500:
        print(f"⚠️  性能偏低（正常应该 >500 samples/sec）")
    else:
        print(f"✅ 性能正常")
    
    print("\n" + "="*80)
    print("常见问题排查")
    print("="*80)
    print("\n1. 如果GPU不可用:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n2. 如果性能偏低:")
    print("   - 检查是否有其他进程占用GPU")
    print("   - 检查GPU温度和功耗")
    print("   - 尝试增加batch_size")
    print("   - 检查是否使用了混合精度训练 (torch.cuda.amp)")
    
    print("\n3. 检查GPU状态:")
    print("   nvidia-smi")
    
    print("\n4. 查看CUDA版本:")
    print("   nvcc --version")
    
    print("\n5. PyTorch CUDA版本:")
    print(f"   {torch.version.cuda}")

if __name__ == "__main__":
    test_gpu_computation()
