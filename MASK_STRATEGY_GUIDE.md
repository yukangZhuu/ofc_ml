# Mask策略配置说明

## 概述

本次修改添加了一个新的可配置选项 `MASK_STRATEGY`，用于控制mask在训练过程中的应用方式。

## 修改内容

### 1. 配置文件修改 (`src/ofc_ml/config.py`)

添加了新的配置选项：

```python
# Mask application strategy
# - "concat": concatenate mask as additional input dimension (original approach)
# - "multiply": multiply mask to spectral features before feeding to network (new approach)
MASK_STRATEGY = "concat"  # one of: {"concat", "multiply"}
```

**说明：**
- `concat`：原始方法，将mask作为额外的输入维度拼接到特征中
- `multiply`：新方法，在输入网络前将mask直接乘到光谱特征上

### 2. 特征预处理修改 (`src/ofc_ml/features.py`)

修改了 `preprocess_features` 函数，添加了 `apply_mask_to_spectra` 参数：

```python
def preprocess_features(train_df, test_df, apply_mask_to_spectra=False):
    # ... 原有代码 ...
    
    if apply_mask_to_spectra:
        print("Applying mask to spectral features before feeding to network...")
        
        # Get indices of spectral features in the transformed array
        spectra_start_idx = 0
        spectra_end_idx = len(spectra_cols)
        
        # Apply mask to spectral features
        train_masks = train_df[mask_cols].values
        test_masks = test_df_processed[mask_cols].values
        
        # Multiply spectral features by mask
        X_train[:, spectra_start_idx:spectra_end_idx] = X_train[:, spectra_start_idx:spectra_end_idx] * train_masks
        X_test[:, spectra_start_idx:spectra_end_idx] = X_test[:, spectra_start_idx:spectra_end_idx] * test_masks
        
        print(f"  Mask applied to {len(spectra_cols)} spectral features")
```

### 3. 模型训练修改 (`src/ofc_ml/model.py`)

#### 3.1 添加辅助函数

```python
def apply_mask_to_features(X, features_df, mask_cols):
    """
    Apply mask to spectral features in transformed feature array.
    """
    print("Applying mask to spectral features...")
    
    # Get masks from original dataframe
    masks = features_df[mask_cols].values
    
    # Apply mask to spectral features (first 95 features are spectral)
    spectra_end_idx = len(mask_cols)
    X[:, :spectra_end_idx] = X[:, :spectra_end_idx] * masks
    
    print(f"  Mask applied to {spectra_end_idx} spectral features")
    
    return X
```

#### 3.2 修改预测逻辑

在 `PyTorchModelWrapper.predict()` 中添加了mask策略判断：

```python
mask_strategy = str(MASK_STRATEGY).lower().strip()

if mask is not None:
    tensor_mask = torch.FloatTensor(mask).to(self.device)
    
    if mask_strategy == "concat":
        # Original approach: concatenate mask as additional input dimension
        # ... 拼接逻辑 ...
    else:
        # New approach: mask already applied to features, just pass through
        preds_offset = self.model(tensor_X)
```

#### 3.3 修改训练循环

在所有训练函数中添加了mask策略判断：
- `train_model()`
- `_train_one_stage()`
- `_train_discriminative_finetune()`
- `train_model_two_stage()`

```python
mask_strategy = str(MASK_STRATEGY).lower().strip()

if mask_strategy == "concat":
    # 拼接mask到输入
    if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
        inputs_for_model = torch.cat([inputs, masks], dim=1)
    elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
        inputs_for_model = torch.cat([inputs, masks], dim=1)
    else:
        inputs_for_model = inputs
else:
    # 不拼接，直接使用输入
    inputs_for_model = inputs
```

#### 3.4 修改模型创建逻辑

在创建模型时根据mask策略调整输入维度：

```python
if mask_strategy == "concat" and FOURIER_KAN_CONCAT_MASK_INPUT:
    input_dim = input_dim + train_masks.shape[1]
    print(f"[fourier_kan] Concatenating mask into input: model_input_dim={input_dim}")
else:
    print(f"[fourier_kan] Using mask strategy: {mask_strategy} (no concatenation)")
```

### 4. 主程序修改 (`main.py`)

添加了命令行参数支持：

```python
p.add_argument("--mask-strategy", choices=["concat", "multiply"], default=None, 
               help="Mask application strategy: 'concat' (concatenate as input dim) or 'multiply' (multiply to spectral features)")
```

在main函数中处理mask策略：

```python
# 处理mask策略参数
if args.mask_strategy is not None:
    cfg.MASK_STRATEGY = args.mask_strategy

# 根据mask策略决定是否在预处理时应用mask
apply_mask_to_spectra = (cfg.MASK_STRATEGY.lower() == "multiply")

print(f"\nMask strategy: {cfg.MASK_STRATEGY}")
if apply_mask_to_spectra:
    print("  -> Will multiply mask to spectral features before feeding to network")
else:
    print("  -> Will concatenate mask as additional input dimension")
```

## 使用方法

### 方法1：通过配置文件

编辑 `src/ofc_ml/config.py`：

```python
# 使用原始方法（拼接mask）
MASK_STRATEGY = "concat"

# 或使用新方法（乘法应用mask）
MASK_STRATEGY = "multiply"
```

### 方法2：通过命令行参数

```bash
# 使用原始方法
python main.py --mask-strategy concat

# 使用新方法
python main.py --mask-strategy multiply
```

## 两种策略的对比

| 特性 | concat (原始方法) | multiply (新方法) |
|------|------------------|-------------------|
| 输入维度 | 特征维度 + 95 | 特征维度 |
| 模型参数量 | 较多 | 较少 |
| mask信息传递 | 显式传递给网络 | 隐式包含在特征中 |
| 计算复杂度 | 较高 | 较低 |
| 物理意义 | 模型学习mask模式 | mask直接过滤非激活通道 |
| 适用场景 | 需要模型理解mask模式 | mask信息直接应用于特征 |

## 技术细节

### concat策略（原始方法）

1. **数据流**：
   ```
   特征 (208维) + Mask (95维) → 拼接 → 模型输入 (303维)
   ```

2. **优点**：
   - 模型可以学习mask的复杂模式
   - mask信息显式传递给网络

3. **缺点**：
   - 增加输入维度，增加模型参数量
   - 计算复杂度较高

### multiply策略（新方法）

1. **数据流**：
   ```
   光谱特征 (95维) × Mask (95维) → 乘法 → 模型输入 (208维)
   ```

2. **优点**：
   - 不增加输入维度
   - 计算效率高
   - 物理意义明确：非激活通道的特征直接置零

3. **缺点**：
   - mask信息隐式包含在特征中
   - 模型无法学习mask的复杂模式

## 注意事项

1. **向后兼容性**：
   - 保留了原有的 `FOURIER_KAN_CONCAT_MASK_INPUT` 和 `HYBRID_FNO_KAN_CONCAT_MASK_INPUT` 配置
   - 只有在 `MASK_STRATEGY == "concat"` 时才会使用这些配置

2. **模型检查点**：
   - 不同mask策略训练的模型不兼容
   - 切换策略时需要重新训练

3. **性能影响**：
   - `multiply`策略可能减少模型参数量，提高训练速度
   - 实际性能需要通过实验验证

## 测试建议

建议进行以下对比实验：

```bash
# 实验1：使用concat策略
python main.py --mask-strategy concat --two-stage

# 实验2：使用multiply策略
python main.py --mask-strategy multiply --two-stage

# 对比两种策略的：
# - 验证集MSE/RMSE
# - 训练时间
# - 模型参数量
# - 测试集性能
```

## 文件修改清单

- [x] `src/ofc_ml/config.py` - 添加MASK_STRATEGY配置
- [x] `src/ofc_ml/features.py` - 修改preprocess_features函数
- [x] `src/ofc_ml/model.py` - 修改所有训练和预测函数
- [x] `main.py` - 添加命令行参数支持

## 总结

本次修改成功实现了mask应用策略的可配置化，用户可以根据实验需求选择最适合的策略。两种策略各有优劣，建议通过实验验证哪种方法在当前任务上表现更好。
