# 两阶段训练快速开始

## 快速运行

### Windows 用户

双击运行：
```
train_two_stage.bat
```

或者命令行：
```bash
python main.py --two-stage
```

### Linux/Mac 用户

```bash
bash train_two_stage.sh
```

或者：
```bash
python main.py --two-stage
```

## 配置说明

在 `src/ofc_ml/config.py` 中可以调整参数：

```python
# 是否默认使用两阶段训练
USE_TWO_STAGE_TRAINING = True

# 预训练参数（COSMOS 数据集）
PRETRAIN_LEARNING_RATE = 0.001      # 预训练学习率
PRETRAIN_BATCH_SIZE = 64            # 批量大小
PRETRAIN_EPOCHS = 500               # 最大轮数
PRETRAIN_EARLY_STOPPING_PATIENCE = 50  # 早停耐心

# 微调参数（Kaggle 数据集）
FINETUNE_LEARNING_RATE = 0.0002     # 微调学习率（更小）
FINETUNE_BATCH_SIZE = 32            # 批量大小（更小）
FINETUNE_EPOCHS = 500               # 最大轮数
FINETUNE_EARLY_STOPPING_PATIENCE = 60  # 早停耐心
```

## 训练流程

```
1. 加载 COSMOS 数据集
   ↓
2. 加载 Kaggle 数据集
   ↓
3. 特征预处理（使用全部数据拟合）
   ↓
4. 阶段 1：COSMOS 预训练
   - 学习通用的 EDFA 增益预测特征
   - 保存预训练模型
   ↓
5. 阶段 2：Kaggle 微调
   - 在目标数据集上精细调整
   - 使用较小学习率
   ↓
6. 评估 & 生成提交文件
```

## 输出文件

训练完成后会生成：

1. **预训练模型**：`models/pretrained_model.pt`
   - 可以重复使用
   - 包含模型权重和配置

2. **提交文件**：`submissions/submission_YYYYMMDD_HHMMSS.csv`
   - 测试集预测结果
   - 可直接提交到 Kaggle

## 性能对比

运行对比脚本查看效果：

```bash
python compare_training_modes.py
```

这会自动运行单阶段和两阶段训练，并输出性能对比。

## 常见问题

### Q: 如何只使用单阶段训练？

A: 修改 `config.py`：
```python
USE_TWO_STAGE_TRAINING = False
```

或运行：
```bash
python main.py  # 不加 --two-stage 参数
```

### Q: 预训练模型保存在哪里？

A: `models/pretrained_model.pt`

### Q: 可以只运行微调阶段吗？

A: 目前框架会自动运行两个阶段。如果想跳过预训练，需要修改代码加载已有模型。

### Q: 内存不够怎么办？

A: 减小 batch size：
```python
PRETRAIN_BATCH_SIZE = 32  # 从 64 改为 32
FINETUNE_BATCH_SIZE = 16  # 从 32 改为 16
```

### Q: 训练时间太长？

A: 减少最大轮数：
```python
PRETRAIN_EPOCHS = 200  # 从 500 改为 200
FINETUNE_EPOCHS = 200
```

## 超参数调优提示

### 如果验证损失不下降

- 增加预训练学习率：`0.002`
- 增加预训练轮数：`1000`

### 如果过拟合

- 增加 dropout：`HYBRID_FNO_KAN_DROPOUT = 0.2`
- 减小模型：`HYBRID_FNO_KAN_HIDDEN_DIMS = [128, 128, 64]`
- 增加 weight decay

### 如果微调效果不好

- 降低微调学习率：`0.0001`
- 增加微调轮数：`1000`
- 使用更小的 batch size：`16`

## 高级用法

### 查看训练日志

训练过程会实时输出：
- 每 20 轮的训练/验证损失
- 学习率变化
- 早停信息

### 修改模型架构

在 `config.py` 中：
```python
MODEL_TYPE = "hybrid_fno_kan"  # 或 "fourier_kan", "mlp"

# 修改网络层数和大小
HYBRID_FNO_KAN_HIDDEN_DIMS = [512, 256, 128, 64]
HYBRID_FNO_KAN_N_FREQUENCIES = 8  # 傅里叶频率数
HYBRID_FNO_KAN_N_SPECTRAL_MODES = 64  # 频域模态数
```

## 预期效果

两阶段训练通常能带来：
- RMSE 降低 5-15%
- 更好的泛化能力
- 更稳定的训练过程

具体效果取决于数据集大小和质量。

## 技术支持

详细文档请参考：
- `TWO_STAGE_TRAINING_README.md`：完整技术文档
- `HYBRID_FNO_KAN_README.md`：模型架构说明

祝训练顺利！🚀
