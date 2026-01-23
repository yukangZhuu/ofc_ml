#!/usr/bin/env python3

import torch
import numpy as np
import time
from pathlib import Path
import sys
import os
from torch.utils.data import DataLoader, TensorDataset

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"

# 添加src目录到Python路径
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 打印路径信息用于调试
print(f"Script dir: {SCRIPT_DIR}")
print(f"Repo root: {REPO_ROOT}")
print(f"Src dir: {SRC_DIR}")
print(f"Python path includes src: {str(SRC_DIR) in sys.path}")
print(f"Current working dir: {os.getcwd()}")
print()

# 尝试导入
try:
    from ofc_ml.network import HybridFNOKANPredictor
    from ofc_ml.model import MaskedMSELoss
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    print(f"Trying alternative import method...")
    
    # 尝试直接导入模块
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
        print("\nPlease run this script from project root directory:")
        print(f"  cd {REPO_ROOT}")
        print(f"  python scripts/test_training_speed.py")
        sys.exit(1)

def test_training_speed():
    print("="*80)
    print("训练速度对比测试")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建模拟数据
    batch_size = 128
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
    
    # 预热
    print("\n预热...")
    for batch in loader:
        inputs, targets, tg, tgt, masks = batch
        inputs, targets, tg, tgt, masks = \
            inputs.to(device), targets.to(device), tg.to(device), tgt.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, masks)
        loss = criterion(outputs, targets, masks)
        loss.backward()
        optimizer.step()
        break
    
    # 测试训练速度
    print("\n开始测试...")
    num_epochs = 5
    warmup_epochs = 1
    
    total_time = 0
    total_samples = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        for inputs, targets, tg, tgt, masks in loader:
            inputs, targets, tg, tgt, masks = \
                inputs.to(device), targets.to(device), tg.to(device), tgt.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()
        
        epoch_time = time.time() - epoch_start
        
        if epoch >= warmup_epochs:
            total_time += epoch_time
            total_samples += num_samples
        
        print(f"Epoch {epoch+1}/{num_epochs}: {epoch_time:.2f}s ({num_samples/epoch_time:.1f} samples/sec)")
    
    avg_time = total_time / (num_epochs - warmup_epochs)
    avg_samples_per_sec = total_samples / total_time
    
    print("\n" + "="*80)
    print("结果（排除预热）")
    print("="*80)
    print(f"平均每epoch时间: {avg_time:.2f}s")
    print(f"平均训练速度: {avg_samples_per_sec:.1f} samples/sec")
    print(f"每batch平均时间: {avg_time/len(loader)*1000:.1f}ms")
    
    if device.type == 'cuda':
        print(f"\nGPU Memory Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        # 性能评估
        if avg_samples_per_sec < 1000:
            print(f"\n⚠️  性能偏低（预期 >1000 samples/sec）")
            print(f"可能原因:")
            print(f"  - DataLoader配置不当（已优化：num_workers={num_workers}, pin_memory={pin_memory}）")
            print(f"  - 频繁的GPU-CPU同步（已修复：移除了训练循环中的.item()调用）")
        else:
            print(f"\n✅ 性能正常！")
    
    print("\n" + "="*80)
    print("优化说明")
    print("="*80)
    print("已应用的优化:")
    print("1. DataLoader: num_workers=4, pin_memory=True（仅GPU）")
    print("2. 训练循环: 移除了频繁的.item()调用，避免GPU-CPU同步")
    print("3. 累积计算: 在循环外计算total_masks，避免重复遍历dataloader")

if __name__ == "__main__":
    test_training_speed()
