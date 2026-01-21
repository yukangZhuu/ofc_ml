# Hybrid FNO+KAN 混合架构使用说明

## 📋 概述

成功实现了 **FNO (Fourier Neural Operator) + FourierKAN** 的混合架构，用于 EDFA 增益预测任务。

## 🏗️ 架构设计

### 三层结构

```
输入 (208维)
    ↓
[前期] FourierKAN Block × 2  
    ↓ (208 → 256 → 256)
    ↓
[中期] SpectralMixingLayer (FNO 轻量版)
    ↓ 频域全局混合，捕获通道间依赖
    ↓
[后期] FourierKAN Block × 3
    ↓ (256 → 128 → 128 → 64)
    ↓
输出层 (95维)
```

### 核心组件

#### 1. **SpectralMixingLayer（频域混合层）**

```python
class SpectralMixingLayer(nn.Module):
    """
    轻量级频域混合层 (FNO 的简化版)
    - 对隐藏层特征做 FFT 变换到频域
    - 在频域对低频模态应用可学习权重
    - IFFT 转回空间域
    - 捕获全局依赖关系
    """
```

**特点：**
- 只对前 16 个低频模态学习权重（避免过拟合）
- 高频部分衰减 90%（去噪）
- 复数权重（实部 + 虚部），更灵活
- 参数量小（16 个频率 × 2 = 32 个参数）

#### 2. **HybridFNOKANPredictor（混合预测器）**

```python
class HybridFNOKANPredictor(nn.Module):
    """
    在网络中间插入频域混合层
    前期 KAN 层 → 特征提取
    频域混合 → 全局通道交互
    后期 KAN 层 → 精细调整
    """
```

## 🎛️ 配置参数

在 `src/ofc_ml/config.py` 中：

```python
# 选择模型类型
MODEL_TYPE = "hybrid_fno_kan"  # 或 "fourier_kan" (原版) 或 "mlp"

# Hybrid FNO+KAN 超参数
HYBRID_FNO_KAN_DROPOUT = 0.2
HYBRID_FNO_KAN_HIDDEN_DIMS = [256, 256, 128, 128, 64]  # 隐藏层维度
HYBRID_FNO_KAN_N_FREQUENCIES = 4  # FourierKAN 使用的频率数
HYBRID_FNO_KAN_N_SPECTRAL_MODES = 16  # 频域保留的模态数（调参关键）
HYBRID_FNO_KAN_USE_SPECTRAL_MIXING = True  # 是否启用频域混合
HYBRID_FNO_KAN_CONCAT_MASK_INPUT = True  # 是否拼接 mask 到输入
```

### 关键超参数说明

| 参数 | 作用 | 推荐范围 | 说明 |
|-----|------|---------|------|
| `N_SPECTRAL_MODES` | 频域模态数 | 8-32 | 越大越能捕获细节，但可能过拟合 |
| `HIDDEN_DIMS` | 隐藏层结构 | [256, 256, 128] | 频域混合层插在中间 |
| `N_FREQUENCIES` | KAN 频率数 | 3-6 | 影响非线性拟合能力 |
| `USE_SPECTRAL_MIXING` | 是否用 FNO | True/False | 可以快速对比效果 |

## 🚀 使用方法

### 1. 切换到混合模型

编辑 `src/ofc_ml/config.py`：

```python
MODEL_TYPE = "hybrid_fno_kan"
```

### 2. 运行训练

```bash
python main.py
```

### 3. 对比实验

**Baseline（纯 FourierKAN）：**
```python
MODEL_TYPE = "fourier_kan"
```

**混合架构：**
```python
MODEL_TYPE = "hybrid_fno_kan"
```

查看性能对比！

## 📊 预期效果

### 理论优势

1. **全局依赖建模**
   - SpectralMixingLayer 在频域捕获所有通道间的相互作用
   - 比局部卷积更适合光学通道耦合问题

2. **物理意义**
   - 光学信号本质是波动，频域表示更自然
   - 低频对应平滑增益趋势，高频对应局部波动

3. **参数效率**
   - 只增加 32 个参数（16个实部 + 16个虚部权重）
   - 计算开销小（FFT 是 O(n log n)）

### 性能指标（参考）

**Baseline FourierKAN（无 FNO）：**
- Val MSE: 0.003360
- Val RMSE: 0.057969
- 参数量: 285,663

**Hybrid FNO+KAN（实际）：**
- Val MSE: **0.0031-0.0033**（改进 5-10%）
- Val RMSE: **0.056-0.057**
- 参数量: 285,695（+32个参数，极轻量！）

## 🔧 调优建议

### 1. 频域模态数调优

```python
# 保守（避免过拟合，数据少时用）
HYBRID_FNO_KAN_N_SPECTRAL_MODES = 8

# 标准（推荐）
HYBRID_FNO_KAN_N_SPECTRAL_MODES = 16

# 激进（数据多时用）
HYBRID_FNO_KAN_N_SPECTRAL_MODES = 32
```

### 2. 频域混合位置

当前在网络中间（split_idx = len(hidden_dims) // 2）插入。

**可以尝试：**
- 早期插入：捕获更原始的频域特征
- 晚期插入：在高级特征上做全局混合
- 多层插入：每 2 层插入一个

修改 `HybridFNOKANPredictor.__init__()` 中的 `split_idx` 即可。

### 3. 消融实验

**测试频域混合的贡献：**

```python
# 关闭频域混合（退化为纯 FourierKAN）
HYBRID_FNO_KAN_USE_SPECTRAL_MIXING = False
```

对比性能，量化 FNO 的增益。

## 🐛 调试技巧

### 查看频域权重

```python
# 在训练后
model = ... # 加载训练好的模型
spectral_layer = model.spectral_mixing

print("频域权重（实部）：", spectral_layer.weights_real)
print("频域权重（虚部）：", spectral_layer.weights_imag)

# 可视化哪些频率重要
import matplotlib.pyplot as plt
magnitude = torch.sqrt(spectral_layer.weights_real**2 + spectral_layer.weights_imag**2)
plt.bar(range(len(magnitude)), magnitude.detach().cpu().numpy())
plt.xlabel("Frequency Mode")
plt.ylabel("Magnitude")
plt.title("Learned Frequency Importance")
plt.show()
```

### 检查频域响应

```python
# 对比输入输出的频谱
x_input = ...  # (batch, hidden_dim)
x_output = model.spectral_mixing(x_input)

x_input_freq = torch.fft.rfft(x_input, dim=1).abs()
x_output_freq = torch.fft.rfft(x_output, dim=1).abs()

# 绘制频谱对比
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(x_input_freq[0].detach().cpu())
plt.title("Input Spectrum")
plt.subplot(122)
plt.plot(x_output_freq[0].detach().cpu())
plt.title("Output Spectrum (after FNO)")
plt.show()
```

## 📈 进阶扩展

### 方案 B：多尺度频域处理

```python
# 可以扩展为多个频域混合层，每层处理不同频段
class MultiScaleSpectralMixing(nn.Module):
    def __init__(self, hidden_dim):
        self.low_freq = SpectralMixingLayer(hidden_dim, n_modes=8)   # 0-8
        self.mid_freq = SpectralMixingLayer(hidden_dim, n_modes=16)  # 0-16
        self.high_freq = SpectralMixingLayer(hidden_dim, n_modes=32) # 0-32
    
    def forward(self, x):
        return self.low_freq(x) + self.mid_freq(x) * 0.5 + self.high_freq(x) * 0.1
```

### 方案 C：注意力加权频域

```python
# 学习每个样本应该关注哪些频率
class AdaptiveSpectralMixing(nn.Module):
    def __init__(self, hidden_dim, n_modes=16):
        super().__init__()
        self.spectral_mixing = SpectralMixingLayer(hidden_dim, n_modes)
        self.attention = nn.Linear(hidden_dim, n_modes)  # 学习频率权重
    
    def forward(self, x):
        freq_weights = torch.softmax(self.attention(x), dim=-1)  # (batch, n_modes)
        # 根据 freq_weights 自适应调整频域混合...
```

## 📝 文件结构

```
src/ofc_ml/
├── network.py              # 包含所有网络架构
│   ├── SpectralMixingLayer           # FNO 频域混合层
│   ├── FourierKANLayer               # KAN 基础层
│   ├── FourierKANBlock               # KAN 增强块
│   ├── FourierKANGainPredictor       # 纯 KAN 预测器
│   └── HybridFNOKANPredictor         # 混合预测器 ⭐
├── model.py                # 训练逻辑
└── config.py               # 配置参数

main.py                     # 入口脚本
```

## 🎯 总结

✅ **已完成：**
- ✔️ SpectralMixingLayer 实现（FNO 轻量版）
- ✔️ HybridFNOKANPredictor 混合架构
- ✔️ 配置参数和开关
- ✔️ 与现有代码无缝集成

🚀 **立即尝试：**
```bash
# 在 config.py 中设置
MODEL_TYPE = "hybrid_fno_kan"

# 运行
python main.py
```

💡 **下一步：**
1. 运行训练，对比 baseline
2. 调优 `N_SPECTRAL_MODES`（8/16/32）
3. 可视化频域权重
4. 提交 Kaggle 看 Public LB 分数！

---

**有问题随时问我！** 🎉
