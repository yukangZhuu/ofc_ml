#!/usr/bin/env python3

import torch
import numpy as np
import time
from pathlib import Path
import sys
import os
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from ofc_ml.network import HybridFNOKANPredictor
    from ofc_ml.model import MaskedMSELoss
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    try:
        import importlib.util
        
        # 导入network.py
        spec = importlib.util.spec_from_file_location("network", SRC_DIR / "ofc_ml" / "network.py")
        network_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(network_module)
        HybridFNOKANPredictor = network_module.HybridFNOKANPredictor
        
        # 导入model.py
        spec = importlib.util.spec_from_file_location("model", SRC_DIR / "ofc_ml" / "model.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        MaskedMSELoss = model_module.MaskedMSELoss
        
        print("✅ Successfully imported using importlib")
    except Exception as e2:
        print(f"❌ Alternative import also failed: {e2}")
        sys.exit(1)

def test_optimizations():
    print("="*80)
    print("训练优化效果测试")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # 创建模拟数据
    batch_size = 256
    num_samples = 10000
    input_dim = 113
    output_dim = 95
    
    print(f"\n数据集大小: {num_samples} samples")
    print(f"Batch size: {batch_size}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    
    # 创建模型
    model = HybridFNOKANPredictor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[256, 256, 128, 128, 64],
        dropout=0.2,
        use_residual=True,
        n_frequencies=4,
        spectral_freq_ratio=0.5,
        use_spectral_mixing=True,
    ).to(device)
    
    # 创建模拟数据
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)
    mask = torch.ones(num_samples, output_dim)
    target_gain = torch.randn(num_samples)
    target_gain_tilt = torch.randn(num_samples)
    
    # 创建dataset和dataloader
    dataset = TensorDataset(X, y, target_gain, target_gain_tilt, mask)
    num_workers = 4 if device.type == 'cuda' else 0
    pin_memory = device.type == 'cuda'
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 测试1: FP32 (无混合精度)
    print("\n" + "="*80)
    print("测试 1: FP32 (无混合精度)")
    print("="*80)
    
    model.train()
    warmup = 5
    epochs = 10
    
    for _ in range(warmup):
        for batch in loader:
            inputs, targets, tg, tgt, masks = batch
            inputs, targets, tg, tgt, masks = \
                inputs.to(device), targets.to(device), tg.to(device), tgt.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            inputs, targets, tg, tgt, masks = batch
            inputs, targets, tg, tgt, masks = \
                inputs.to(device), targets.to(device), tg.to(device), tgt.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize()
    elapsed_fp32 = time.time() - start
    avg_time_fp32 = elapsed_fp32 / epochs
    samples_per_sec_fp32 = num_samples / avg_time_fp32
    
    print(f"总时间: {elapsed_fp32:.2f}s")
    print(f"平均每epoch: {avg_time_fp32:.2f}s")
    print(f"训练速度: {samples_per_sec_fp32:.1f} samples/sec")
    
    # 测试2: FP16 (混合精度)
    print("\n" + "="*80)
    print("测试 2: FP16 (混合精度)")
    print("="*80)
    
    from torch.cuda.amp import autocast, GradScaler
    
    model.train()
    scaler = GradScaler()
    
    for _ in range(warmup):
        for batch in loader:
            inputs, targets, tg, tgt, masks = batch
            inputs, targets, tg, tgt, masks = \
                inputs.to(device), targets.to(device), tg.to(device), tgt.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs, masks)
                loss = criterion(outputs, targets, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            inputs, targets, tg, tgt, masks = batch
            inputs, targets, tg, tgt, masks = \
                inputs.to(device), targets.to(device), tg.to(device), tgt.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs, masks)
                loss = criterion(outputs, targets, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
    
    torch.cuda.synchronize()
    elapsed_fp16 = time.time() - start
    avg_time_fp16 = elapsed_fp16 / epochs
    samples_per_sec_fp16 = num_samples / avg_time_fp16
    
    print(f"总时间: {elapsed_fp16:.2f}s")
    print(f"平均每epoch: {avg_time_fp16:.2f}s")
    print(f"训练速度: {samples_per_sec_fp16:.1f} samples/sec")
    
    # 对比结果
    print("\n" + "="*80)
    print("性能对比")
    print("="*80)
    print(f"FP32: {avg_time_fp32:.2f}s/epoch ({samples_per_sec_fp32:.1f} samples/sec)")
    print(f"FP16: {avg_time_fp16:.2f}s/epoch ({samples_per_sec_fp16:.1f} samples/sec)")
    
    speedup = avg_time_fp32 / avg_time_fp16
    print(f"\n加速比: {speedup:.2f}x")
    
    if speedup > 1.2:
        print("✅ 混合精度训练效果显著！")
    elif speedup > 1.0:
        print("⚠️  混合精度训练有轻微提升")
    else:
        print("❌ 混合精度训练没有提升，可能需要调整")
    
    # GPU内存使用
    if device.type == 'cuda':
        print(f"\nGPU Memory Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    print("\n" + "="*80)
    print("优化建议")
    print("="*80)
    print("1. Batch Size: 当前256，可以尝试512（如果GPU内存足够）")
    print("2. 混合精度: 已启用，可节省内存并加速")
    print("3. 验证频率: 预训练每2个epoch验证一次")
    print("4. DataLoader: num_workers=4, pin_memory=True")
    print("\n预期优化效果:")
    print("- Batch Size翻倍: ~2x加速")
    print("- 混合精度: ~1.2-1.5x加速")
    print("- 减少验证: ~1.1x加速")
    print("- 综合加速: ~2.5-3x加速")
    print(f"\n预期每epoch时间: {6.55 / 2.5:.2f}s (从6.55s优化)")

if __name__ == "__main__":
    test_optimizations()
