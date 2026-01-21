# 两阶段训练框架说明

## 概述

本框架实现了一个两阶段的迁移学习策略：

1. **阶段 1：预训练（Pretrain）** - 使用 COSMOS 数据集进行预训练
2. **阶段 2：微调（Finetune）** - 使用 Kaggle 数据集进行微调

## 设计思路

### 为什么使用两阶段训练？

1. **数据规模差异**：COSMOS 数据集通常比 Kaggle 训练集大，可以提供更多的训练样本
2. **迁移学习**：先在大数据集上学习通用特征，再在目标数据集上微调
3. **泛化能力**：预训练可以帮助模型学习更鲁棒的特征表示
4. **防止过拟合**：在较小的目标数据集上使用较低的学习率微调，减少过拟合风险

### 训练策略

#### 预训练阶段（COSMOS）
- 学习率：`0.001`（相对较大）
- Batch size：`64`
- Early stopping patience：`50`
- 目标：学习 EDFA 增益预测的通用特征

#### 微调阶段（Kaggle）
- 学习率：`0.0002`（较小，保留预训练的知识）
- Batch size：`32`（更小的 batch，更精细的更新）
- Early stopping patience：`60`
- 目标：适应目标数据集的特定分布

## 使用方法

### 1. 配置文件设置

在 `src/ofc_ml/config.py` 中：

```python
# 启用两阶段训练
USE_TWO_STAGE_TRAINING = True

# 预训练参数
PRETRAIN_LEARNING_RATE = 0.001
PRETRAIN_WEIGHT_DECAY = 1e-4
PRETRAIN_BATCH_SIZE = 64
PRETRAIN_EPOCHS = 500
PRETRAIN_EARLY_STOPPING_PATIENCE = 50

# 微调参数
FINETUNE_LEARNING_RATE = 0.0002
FINETUNE_WEIGHT_DECAY = 1e-5
FINETUNE_BATCH_SIZE = 32
FINETUNE_EPOCHS = 500
FINETUNE_EARLY_STOPPING_PATIENCE = 60
```

### 2. 运行训练

#### 方法 A：使用配置文件中的设置

```bash
python main.py
```

#### 方法 B：使用命令行参数强制启用两阶段训练

```bash
python main.py --two-stage
```

#### 方法 C：使用单阶段训练（原始方法）

```bash
# 修改 config.py 中的 USE_TWO_STAGE_TRAINING = False
# 或者不使用 --two-stage 参数且配置文件中为 False
python main.py
```

## 训练流程

### 详细步骤

1. **数据准备**
   - 分别加载 COSMOS 和 Kaggle 数据集
   - 使用所有数据（COSMOS + Kaggle）拟合特征预处理器
   - 分别对两个数据集进行预处理

2. **预训练（Stage 1）**
   - 在 COSMOS 数据集上训练模型
   - 使用较大的学习率和批量大小
   - 保存预训练模型到 `models/pretrained_model.pt`

3. **微调（Stage 2）**
   - 加载预训练的模型权重
   - 在 Kaggle 数据集上继续训练
   - 使用较小的学习率精细调整

4. **评估和预测**
   - 在 Kaggle 验证集上评估最终模型
   - 在测试集上生成预测
   - 创建提交文件

## 输出文件

- `models/pretrained_model.pt`：预训练模型检查点
- `submissions/submission_YYYYMMDD_HHMMSS.csv`：测试集预测结果

## 超参数调优建议

### 预训练阶段

如果预训练损失下降缓慢：
- 增加学习率（例如 `0.002`）
- 增加 batch size（例如 `128`）

如果预训练过拟合：
- 增加 dropout（在 config 中修改 `HYBRID_FNO_KAN_DROPOUT`）
- 减少模型复杂度
- 增加 weight decay

### 微调阶段

如果微调后性能下降：
- 降低学习率（例如 `0.0001`）
- 减少训练轮数
- 使用更小的 batch size

如果微调不足：
- 增加学习率（例如 `0.0005`）
- 增加训练轮数
- 减小 early stopping patience

## 监控训练过程

训练过程中会输出：

```
================================================================================
STAGE 1: PRETRAINING ON COSMOS
================================================================================
Learning rate: 0.001
Weight decay: 0.0001
Max epochs: 500
Early stopping patience: 50

Epoch 20/500 - Train Loss: 0.123456 - Val Loss: 0.234567 - LR: 0.001000
...
Early stopping at epoch 123
STAGE 1: PRETRAINING ON COSMOS completed. Best val loss: 0.234567

Pretrained model saved to: models/pretrained_model.pt

================================================================================
STAGE 2: FINETUNING ON KAGGLE
================================================================================
Learning rate: 0.0002
Weight decay: 1e-05
Max epochs: 500
Early stopping patience: 60

Epoch 20/500 - Train Loss: 0.098765 - Val Loss: 0.123456 - LR: 0.000200
...
```

## 架构兼容性

该两阶段训练框架支持所有模型架构：

- `mlp`：SimpleGainPredictor
- `fourier_kan`：FourierKANGainPredictor
- `hybrid_fno_kan`：HybridFNOKANPredictor（推荐）

在 `config.py` 中设置：

```python
MODEL_TYPE = "hybrid_fno_kan"
```

## 高级功能

### 从预训练模型恢复

如果想要仅使用已有的预训练模型进行微调，可以：

1. 确保 `models/pretrained_model.pt` 存在
2. 在代码中添加加载逻辑
3. 直接运行微调阶段

### 自定义训练策略

可以在 `src/ofc_ml/model.py` 中的 `_train_one_stage` 函数中自定义：

- 学习率调度策略
- 损失函数
- 优化器选择
- 梯度裁剪策略

## 注意事项

1. **数据一致性**：确保 COSMOS 和 Kaggle 数据集的特征列完全一致
2. **内存使用**：两阶段训练会同时加载两个数据集，注意内存占用
3. **训练时间**：两阶段训练比单阶段训练时间更长，但通常效果更好
4. **模型保存**：预训练模型会保存，可以用于后续实验

## 性能对比

建议同时运行单阶段和两阶段训练，对比效果：

```bash
# 单阶段（baseline）
python main.py  # 设置 USE_TWO_STAGE_TRAINING = False

# 两阶段
python main.py  # 设置 USE_TWO_STAGE_TRAINING = True
```

对比验证集的 RMSE/MAE 来评估两阶段训练的效果。

## 故障排除

### 问题：预训练模型无法加载

- 检查 `models/` 目录是否存在
- 确认模型架构参数一致

### 问题：内存不足

- 减小 batch size
- 减少数据集大小
- 使用梯度累积

### 问题：训练不收敛

- 调整学习率
- 检查数据预处理是否正确
- 验证标签计算（baseline + offset）

## 总结

两阶段训练框架为模型提供了更好的初始化和泛化能力。通过合理的超参数设置和训练策略，可以充分利用 COSMOS 和 Kaggle 两个数据集的优势，提升最终的预测性能。
