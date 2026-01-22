# 预训练模型使用指南

本指南说明如何从预训练模型基础上进行迁移学习/微调训练。

## 快速开始

### 1. 从预训练模型继续训练（推荐）

如果 `models/pretrained_model.pt` 文件已经存在，您可以直接加载它并跳过预训练阶段：

```bash
# 方式1：使用配置文件（默认行为）
# 在 config.py 中设置 LOAD_PRETRAINED_MODEL = True
python main.py

# 方式2：使用命令行参数（覆盖配置）
python main.py --load-pretrained
```

**优点**：
- 节省时间：跳过预训练阶段，直接进入微调
- 保留之前训练的知识
- 适合迭代式实验和调优

### 2. 从头开始完整训练

如果您想重新训练整个模型（忽略已有的预训练模型）：

```bash
# 方式1：在 config.py 中设置 LOAD_PRETRAINED_MODEL = False
python main.py

# 方式2：使用命令行参数
python main.py --no-load-pretrained
```

## 配置说明

在 `src/ofc_ml/config.py` 中：

```python
# 是否从已有的预训练模型加载并跳过预训练阶段
LOAD_PRETRAINED_MODEL = True  # 默认为 True

# 预训练模型保存路径
PRETRAIN_MODEL_PATH = PROJECT_ROOT / "models" / "pretrained_model.pt"
```

## 工作流程

### 完整两阶段训练流程

```
┌─────────────────────────────────────┐
│  检查 LOAD_PRETRAINED_MODEL         │
│  和 pretrained_model.pt 是否存在    │
└─────────────┬───────────────────────┘
              │
      ┌───────┴────────┐
      │  是否加载？      │
      └───┬────────┬───┘
          │        │
    ✓ 是  │        │ ✗ 否
          │        │
          ▼        ▼
    ┌─────────┐  ┌──────────────────┐
    │ 加载模型 │  │ Stage 1: 预训练   │
    │ (跳过   │  │ (COSMOS 数据集)   │
    │ Stage 1)│  │                  │
    └────┬────┘  └────────┬─────────┘
         │                │
         │                ▼
         │          ┌─────────────┐
         │          │ 保存预训练   │
         │          │ 模型到文件   │
         │          └──────┬──────┘
         │                 │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │ Stage 2: 微调    │
         │ (Kaggle 数据集)  │
         └─────────┬────────┘
                   ▼
         ┌─────────────────┐
         │  生成预测结果    │
         └─────────────────┘
```

## 预训练模型文件结构

预训练模型 `.pt` 文件包含以下信息：

```python
{
    'model_state_dict': ...,     # 模型权重
    'model_type': 'hybrid_fno_kan',  # 模型类型
    'input_dim': 128,            # 输入维度
    'output_dim': 96,            # 输出维度
    'pretrain_loss': 0.0123,     # 预训练验证损失
}
```

## 常见使用场景

### 场景1：第一次训练
```bash
# 完整训练（预训练 + 微调）
python main.py --no-load-pretrained
```
结果：会生成 `models/pretrained_model.pt`

### 场景2：调整微调参数后重新训练
```bash
# 从预训练模型开始，只重新微调
python main.py --load-pretrained
```
这样可以快速实验不同的微调超参数。

### 场景3：更换模型架构
```bash
# 在 config.py 中修改 MODEL_TYPE
# 然后从头训练新模型
python main.py --no-load-pretrained
```

### 场景4：增加更多预训练数据
```bash
# 添加新的 COSMOS 数据后，重新预训练
python main.py --no-load-pretrained
```

## 高级技巧

### 1. 保存多个预训练模型

您可以手动保存不同版本的预训练模型：

```bash
# 训练后，复制预训练模型
cp models/pretrained_model.pt models/pretrained_model_v1.pt
cp models/pretrained_model.pt models/pretrained_model_v2.pt
```

然后在 `config.py` 中切换：
```python
# 使用特定版本
PRETRAIN_MODEL_PATH = PROJECT_ROOT / "models" / "pretrained_model_v1.pt"
```

### 2. 自定义加载逻辑

如果需要更复杂的加载逻辑，可以修改 `src/ofc_ml/model.py` 中的 `train_model_two_stage` 函数。

## 命令行参数总结

| 参数 | 说明 | 示例 |
|------|------|------|
| `--load-pretrained` | 强制加载预训练模型 | `python main.py --load-pretrained` |
| `--no-load-pretrained` | 强制不加载，从头训练 | `python main.py --no-load-pretrained` |
| `--two-stage` | 启用两阶段训练模式 | `python main.py --two-stage` |
| `--dataset-use` | 选择数据集 | `python main.py --dataset-use both` |

## 注意事项

1. **模型兼容性**：确保预训练模型的架构与当前配置一致（MODEL_TYPE、输入输出维度等）
2. **数据预处理**：加载预训练模型时，确保使用相同的数据预处理流程
3. **超参数调整**：微调时通常使用更小的学习率（FINETUNE_LEARNING_RATE）
4. **设备一致性**：确保训练和加载时使用相同的设备（CPU/CUDA）

## 故障排除

### 问题1：加载预训练模型失败

```
✗ Failed to load pretrained model: ...
```

**解决方案**：
- 检查 `models/pretrained_model.pt` 是否存在
- 检查模型架构配置是否与训练时一致
- 尝试使用 `--no-load-pretrained` 重新训练

### 问题2：模型维度不匹配

```
RuntimeError: Error(s) in loading state_dict...
```

**解决方案**：
- 确认 MODEL_TYPE 与预训练模型一致
- 检查 FOURIER_KAN_CONCAT_MASK_INPUT 等配置
- 重新训练新模型

## 性能对比

| 训练模式 | 预训练时间 | 微调时间 | 总时间 | 适用场景 |
|---------|-----------|---------|--------|---------|
| 完整训练 | ~2-4小时 | ~1-2小时 | ~3-6小时 | 首次训练 |
| 加载预训练 | 跳过 | ~1-2小时 | ~1-2小时 | 调整微调参数 |

## 最佳实践

1. **保存检查点**：定期备份 `models/pretrained_model.pt`
2. **版本管理**：为不同实验保存不同版本的预训练模型
3. **文档记录**：记录每个预训练模型的训练配置和性能指标
4. **迭代实验**：使用预训练模型快速迭代微调参数

---

更新日期：2026-01-22
