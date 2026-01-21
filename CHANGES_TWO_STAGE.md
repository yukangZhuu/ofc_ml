# 两阶段训练框架 - 修改总结

## 修改概览

为了实现两阶段训练（COSMOS 预训练 + Kaggle 微调），对以下文件进行了修改和新增：

## 修改的文件

### 1. `src/ofc_ml/config.py`
**修改内容**：添加两阶段训练相关配置

```python
# 新增配置
USE_TWO_STAGE_TRAINING = True
PRETRAIN_LEARNING_RATE = 0.001
PRETRAIN_WEIGHT_DECAY = 1e-4
PRETRAIN_BATCH_SIZE = 64
PRETRAIN_EPOCHS = 500
PRETRAIN_EARLY_STOPPING_PATIENCE = 50
FINETUNE_LEARNING_RATE = 0.0002
FINETUNE_WEIGHT_DECAY = 1e-5
FINETUNE_BATCH_SIZE = 32
FINETUNE_EPOCHS = 500
FINETUNE_EARLY_STOPPING_PATIENCE = 60
PRETRAIN_MODEL_PATH = PROJECT_ROOT / "models" / "pretrained_model.pt"
```

**其他修改**：
- `HYBRID_FNO_KAN_DROPOUT = 0.1`（从 0.2 改为 0.1）
- `HYBRID_FNO_KAN_N_FREQUENCIES = 8`（从 4 改为 8）

### 2. `src/ofc_ml/model.py`
**修改内容**：

1. **新增导入**：导入两阶段训练相关的配置参数

2. **修复 Bug**：在 `PyTorchModelWrapper.predict()` 中添加对 `hybrid_fno_kan` 模型的 mask 拼接支持
   ```python
   elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
       tensor_X = torch.cat([tensor_X, tensor_mask], dim=1)
   ```

3. **新增函数 `_train_one_stage()`**：通用的单阶段训练函数
   - 支持预训练和微调两个阶段
   - 可配置学习率、批量大小、轮数等
   - 包含完整的训练循环和早停机制

4. **新增函数 `train_model_two_stage()`**：两阶段训练主函数
   - 分别加载和预处理 COSMOS 和 Kaggle 数据
   - 调用 `_train_one_stage()` 进行预训练
   - 保存预训练模型
   - 调用 `_train_one_stage()` 进行微调
   - 在 Kaggle 验证集上评估最终模型

### 3. `src/ofc_ml/data.py`
**修改内容**：

新增函数 `load_data_separate()`：
- 分别加载 COSMOS 和 Kaggle 数据集
- 验证两个数据集的 schema 一致性
- 返回五个数据框：cosmos_features, cosmos_labels, kaggle_features, kaggle_labels, test_features

### 4. `main.py`
**修改内容**：

1. **新增导入**：
   ```python
   import pandas as pd
   from ofc_ml.data import load_data_separate
   from ofc_ml.model import train_model_two_stage
   ```

2. **修改命令行参数**：
   ```python
   p.add_argument("--two-stage", action="store_true", help="Use two-stage training")
   ```

3. **重写 `main()` 函数**：
   - 根据 `--two-stage` 参数或 `USE_TWO_STAGE_TRAINING` 配置选择训练模式
   - 两阶段模式：调用 `load_data_separate()` 和 `train_model_two_stage()`
   - 单阶段模式：保持原有逻辑不变

## 新增的文件

### 文档类

1. **`TWO_STAGE_TRAINING_README.md`**
   - 完整的两阶段训练技术文档
   - 设计思路、使用方法、超参数调优建议
   - 故障排除指南

2. **`QUICK_START_TWO_STAGE.md`**
   - 快速开始指南
   - 常见问题解答
   - 简化的配置说明

3. **`CHANGES_TWO_STAGE.md`**（本文件）
   - 所有修改的详细列表
   - 架构说明

### 脚本类

4. **`train_two_stage.sh`**
   - Linux/Mac 启动脚本

5. **`train_two_stage.bat`**
   - Windows 启动脚本

6. **`compare_training_modes.py`**
   - 自动对比单阶段和两阶段训练效果的脚本

## 架构说明

### 训练流程

```
main.py
  ├─ 判断是否使用两阶段训练
  │
  ├─ [两阶段模式]
  │   ├─ load_data_separate()        # data.py
  │   │   ├─ 加载 COSMOS 数据
  │   │   └─ 加载 Kaggle 数据
  │   │
  │   ├─ preprocess_features()       # features.py
  │   │   └─ 使用全部数据拟合 preprocessor
  │   │
  │   └─ train_model_two_stage()     # model.py
  │       ├─ 准备 COSMOS 数据集
  │       ├─ 准备 Kaggle 数据集
  │       ├─ 创建模型
  │       ├─ _train_one_stage() [预训练]
  │       │   └─ 在 COSMOS 上训练
  │       ├─ 保存预训练模型
  │       ├─ _train_one_stage() [微调]
  │       │   └─ 在 Kaggle 上微调
  │       └─ 返回最终模型
  │
  └─ [单阶段模式]
      ├─ load_data()                 # data.py
      ├─ preprocess_features()       # features.py
      └─ train_model()               # model.py
```

### 关键设计决策

1. **预处理器共享**：使用 COSMOS + Kaggle 全部数据拟合 preprocessor，确保特征分布一致

2. **Baseline + Offset 架构**：两个阶段都使用相同的 baseline+offset 预测方式

3. **学习率策略**：
   - 预训练：较大学习率（0.001）快速学习
   - 微调：较小学习率（0.0002）精细调整

4. **批量大小策略**：
   - 预训练：较大 batch（64）保证稳定
   - 微调：较小 batch（32）更精细的更新

5. **模型保存**：只保存预训练模型，方便后续实验

## 兼容性

### 向后兼容
- 保留了原有的单阶段训练功能
- 通过 `USE_TWO_STAGE_TRAINING = False` 可以回退到原始行为
- 所有原有的配置参数都保持不变

### 模型架构支持
- ✅ MLP
- ✅ FourierKAN
- ✅ HybridFNOKAN（推荐）

### 数据集支持
- ✅ COSMOS 数据集
- ✅ Kaggle 数据集
- ✅ 分别加载和处理

## 使用示例

### 示例 1：使用两阶段训练

```bash
# 方法 1：使用配置文件
# 在 config.py 中设置 USE_TWO_STAGE_TRAINING = True
python main.py

# 方法 2：使用命令行参数
python main.py --two-stage

# 方法 3：使用启动脚本
python train_two_stage.bat  # Windows
bash train_two_stage.sh     # Linux/Mac
```

### 示例 2：使用单阶段训练

```bash
# 在 config.py 中设置 USE_TWO_STAGE_TRAINING = False
python main.py
```

### 示例 3：对比两种模式

```bash
python compare_training_modes.py
```

## 测试建议

### 1. 功能测试
- [ ] 运行两阶段训练，确保正常完成
- [ ] 验证预训练模型保存成功
- [ ] 检查提交文件格式正确

### 2. 性能测试
- [ ] 对比单阶段和两阶段的验证集 RMSE
- [ ] 记录训练时间
- [ ] 检查 GPU 内存使用

### 3. 超参数测试
- [ ] 测试不同的预训练学习率
- [ ] 测试不同的微调学习率
- [ ] 测试不同的 batch size 组合

## 已知限制

1. **内存占用**：两阶段训练会同时加载两个数据集，需要更多内存
2. **训练时间**：比单阶段训练时间更长（约 1.5-2 倍）
3. **模型保存**：目前只保存预训练模型，不保存微调后的模型（可以添加）

## 未来改进方向

1. **渐进式微调**：可以尝试逐层解冻微调
2. **学习率预热**：在微调开始时使用学习率预热
3. **数据增强**：在预训练阶段加入数据增强
4. **多阶段训练**：可以扩展为 3+ 阶段训练
5. **自动超参数调优**：使用 Optuna 等工具自动搜索最佳超参数

## Bug 修复

修复了 `model.py` 中的一个 bug：
- **问题**：在使用 `hybrid_fno_kan` 模型且 `HYBRID_FNO_KAN_CONCAT_MASK_INPUT = True` 时，训练时会拼接 mask，但预测时不拼接，导致维度不匹配
- **修复**：在 `PyTorchModelWrapper.predict()` 中添加对 `hybrid_fno_kan` 的判断和处理

## 版本信息

- 创建日期：2026-01-21
- 框架版本：1.0
- Python 要求：3.8+
- PyTorch 要求：1.10+

## 联系方式

如有问题或建议，请查看文档或修改代码。

---

**祝训练顺利！** 🚀
