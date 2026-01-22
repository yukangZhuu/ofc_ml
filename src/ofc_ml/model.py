import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from pathlib import Path

from .config import (
    RANDOM_STATE,
    TEST_SIZE,
    WANDB_PROJECT,
    WANDB_ENTITY,
    WANDB_MODE,
    MODEL_TYPE,
    DROPOUT,
    HIDDEN_DIMS,
    FOURIER_KAN_DROPOUT,
    FOURIER_KAN_HIDDEN_DIMS,
    FOURIER_KAN_N_FREQUENCIES,
    FOURIER_KAN_CONCAT_MASK_INPUT,
    HYBRID_FNO_KAN_DROPOUT,
    HYBRID_FNO_KAN_HIDDEN_DIMS,
    HYBRID_FNO_KAN_N_FREQUENCIES,
    HYBRID_FNO_KAN_N_SPECTRAL_MODES,
    HYBRID_FNO_KAN_USE_SPECTRAL_MIXING,
    HYBRID_FNO_KAN_CONCAT_MASK_INPUT,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    DEVICE,
    USE_TWO_STAGE_TRAINING,
    LOAD_PRETRAINED_MODEL,
    PRETRAIN_LEARNING_RATE,
    PRETRAIN_WEIGHT_DECAY,
    PRETRAIN_BATCH_SIZE,
    PRETRAIN_EPOCHS,
    PRETRAIN_EARLY_STOPPING_PATIENCE,
    FINETUNE_LEARNING_RATE,
    FINETUNE_WEIGHT_DECAY,
    FINETUNE_BATCH_SIZE,
    FINETUNE_EPOCHS,
    FINETUNE_EARLY_STOPPING_PATIENCE,
    PRETRAIN_MODEL_PATH,
    DISCRIMINATIVE_LR_DECAY,
)
from .network import SimpleGainPredictor, FourierKANGainPredictor, HybridFNOKANPredictor, OFCDataset, TargetNormalizer, compute_baseline_gain

class PyTorchModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, X, target_gain, target_gain_tilt, mask=None):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).to(self.device)
            
            if mask is not None:
                tensor_mask = torch.FloatTensor(mask).to(self.device)
                # If the model was trained with mask concatenated into inputs, do the same at inference.
                model_type = str(MODEL_TYPE).lower().strip()
                if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                    tensor_X = torch.cat([tensor_X, tensor_mask], dim=1)
                elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                    tensor_X = torch.cat([tensor_X, tensor_mask], dim=1)
                preds_offset = self.model(tensor_X, tensor_mask)
            else:
                preds_offset = self.model(tensor_X)
            
            preds_offset = preds_offset.cpu().numpy()
            
            # Compute baseline
            baseline = np.array([
                compute_baseline_gain(tg, tgt) 
                for tg, tgt in zip(target_gain, target_gain_tilt)
            ])
            
            # Final prediction: baseline + offset
            preds = baseline + preds_offset
            
            if mask is not None:
                preds = preds * mask
            
            return preds

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        
    def forward(self, predictions, targets, mask):
        mask = mask.to(predictions.device)
        squared_diff = (predictions - targets) ** 2
        masked_squared_diff = squared_diff * mask
        loss = masked_squared_diff.sum() / mask.sum()
        return loss


def get_layer_groups(model):
    """
    将模型分成多个层组，用于分层学习率设置
    返回：层组列表，从底层到顶层
    """
    layer_groups = []
    
    # 获取所有命名的子模块
    named_modules = list(model.named_children())
    
    if len(named_modules) == 0:
        # 如果没有子模块，返回整个模型
        return [list(model.parameters())]
    
    # 将每个主要模块作为一个组
    for name, module in named_modules:
        params = list(module.parameters())
        if len(params) > 0:
            layer_groups.append(params)
    
    return layer_groups


def setup_discriminative_lr(model, base_lr, decay_factor=0.95):
    """
    设置判别式学习率：不同层使用不同的学习率
    底层使用更小的学习率，顶层使用更大的学习率
    
    Args:
        model: PyTorch模型
        base_lr: 顶层的基础学习率
        decay_factor: 学习率衰减因子（每往底层走一层，学习率乘以这个因子）
    
    Returns:
        参数组列表，可直接传给optimizer
    """
    layer_groups = get_layer_groups(model)
    num_groups = len(layer_groups)
    
    param_groups = []
    for i, params in enumerate(layer_groups):
        # 从底层到顶层，底层的i较小，所以lr较小
        # 顶层的i较大（接近num_groups-1），所以lr较大
        lr_multiplier = decay_factor ** (num_groups - 1 - i)
        layer_lr = base_lr * lr_multiplier
        param_groups.append({
            'params': params,
            'lr': layer_lr
        })
        print(f"  Layer group {i}: lr = {layer_lr:.6f} (multiplier: {lr_multiplier:.4f})")
    
    return param_groups


def train_model(X_train, y_train, preprocessor, train_features, mask_cols):
    print("Training simplified Neural Network model with baseline+offset approach...")
    
    if torch.cuda.is_available():
        device = torch.device(str(DEVICE))
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    target_gain_values = train_features['target_gain'].values
    target_gain_tilt_values = train_features['target_gain_tilt'].values
    
    print(f"\nComputing baseline gains...")
    baseline_gains = np.array([
        compute_baseline_gain(tg, tgt) 
        for tg, tgt in zip(target_gain_values, target_gain_tilt_values)
    ])
    
    print(f"Baseline gain shape: {baseline_gains.shape}")
    print(f"  Mean: {baseline_gains.mean():.4f}")
    print(f"  Std: {baseline_gains.std():.4f}")
    print(f"  Min: {baseline_gains.min():.4f}")
    print(f"  Max: {baseline_gains.max():.4f}")
    
    print(f"\nComputing offset values (only for active channels)...")
    train_masks_array = train_features[mask_cols].values
    
    offset_values = (y_train - baseline_gains) * train_masks_array
    
    print(f"Offset statistics:")
    print(f"  Mean: {offset_values.mean():.6f}")
    print(f"  Std: {offset_values.std():.6f}")
    print(f"  Min: {offset_values.min():.6f}")
    print(f"  Max: {offset_values.max():.6f}")
    
    train_masks = train_features[mask_cols].values
    
    X_tr, X_val, y_tr, y_val, mask_tr, mask_val, tg_tr, tg_val, tgt_tr, tgt_val = train_test_split(
        X_train, offset_values, train_masks, 
        target_gain_values, target_gain_tilt_values,
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    train_dataset = OFCDataset(X_tr, y_tr, tg_tr, tgt_tr, mask_tr)
    val_dataset = OFCDataset(X_val, y_val, tg_val, tgt_val, mask_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model_type = str(MODEL_TYPE).lower().strip()
    if model_type == "mlp":
        model = SimpleGainPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT
        ).to(device)
    elif model_type in {"fourier_kan", "simple_kan"}:
        # Optionally include the 95-dim activation mask as part of the input features.
        # This helps the model generalize across different channel activation patterns.
        if FOURIER_KAN_CONCAT_MASK_INPUT:
            input_dim = input_dim + train_masks.shape[1]
            print(f"[fourier_kan] Concatenating mask into input: model_input_dim={input_dim}")
        model = FourierKANGainPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=FOURIER_KAN_HIDDEN_DIMS,
            dropout=FOURIER_KAN_DROPOUT,
            use_residual=True,
            n_frequencies=FOURIER_KAN_N_FREQUENCIES,
        ).to(device)
    elif model_type == "hybrid_fno_kan":
        # Hybrid FNO + FourierKAN architecture
        if HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
            input_dim = input_dim + train_masks.shape[1]
            print(f"[hybrid_fno_kan] Concatenating mask into input: model_input_dim={input_dim}")
        model = HybridFNOKANPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=HYBRID_FNO_KAN_HIDDEN_DIMS,
            dropout=HYBRID_FNO_KAN_DROPOUT,
            use_residual=True,
            n_frequencies=HYBRID_FNO_KAN_N_FREQUENCIES,
            n_spectral_modes=HYBRID_FNO_KAN_N_SPECTRAL_MODES,
            use_spectral_mixing=HYBRID_FNO_KAN_USE_SPECTRAL_MIXING,
        ).to(device)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE={MODEL_TYPE!r}. Use 'mlp', 'fourier_kan', or 'hybrid_fno_kan'.")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    criterion = MaskedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    
    epochs = 1000
    best_val_loss = float('inf')
    best_model_state = None
    patience = EARLY_STOPPING_PATIENCE
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets, target_gain, target_gain_tilt, masks in train_loader:
            inputs, targets, target_gain, target_gain_tilt, masks = \
                inputs.to(device), targets.to(device), target_gain.to(device), \
                target_gain_tilt.to(device), masks.to(device)
            
            optimizer.zero_grad()
            if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                inputs_for_model = torch.cat([inputs, masks], dim=1)
            elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                inputs_for_model = torch.cat([inputs, masks], dim=1)
            else:
                inputs_for_model = inputs
            outputs = model(inputs_for_model, masks)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * masks.sum().item()
            
        total_train_masks = sum(masks.sum().item() for _, _, _, _, masks in train_loader)
        train_loss /= total_train_masks
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, target_gain, target_gain_tilt, masks in val_loader:
                inputs, targets, target_gain, target_gain_tilt, masks = \
                    inputs.to(device), targets.to(device), target_gain.to(device), \
                    target_gain_tilt.to(device), masks.to(device)
                if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                    inputs_for_model = torch.cat([inputs, masks], dim=1)
                elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                    inputs_for_model = torch.cat([inputs, masks], dim=1)
                else:
                    inputs_for_model = inputs
                outputs = model(inputs_for_model, masks)
                loss = criterion(outputs, targets, masks)
                val_loss += loss.item() * masks.sum().item()
        
        total_val_masks = sum(masks.sum().item() for _, _, _, _, masks in val_loader)
        val_loss /= total_val_masks
        
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.6f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    wrapper = PyTorchModelWrapper(model, device)
    y_pred = wrapper.predict(X_val, tg_val, tgt_val, mask_val)
    
    y_pred_masked = y_pred * mask_val
    
    y_val_denorm = y_val + np.array([
        compute_baseline_gain(tg, tgt) 
        for tg, tgt in zip(tg_val, tgt_val)
    ])
    y_val_masked = y_val_denorm * mask_val
    
    non_zero_mask = mask_val > 0
    mse = mean_squared_error(y_val_masked[non_zero_mask], y_pred_masked[non_zero_mask])
    mae = mean_absolute_error(y_val_masked[non_zero_mask], y_pred_masked[non_zero_mask])
    rmse = np.sqrt(mse)
    
    print(f"Final Validation MSE: {mse:.6f}")
    print(f"Final Validation RMSE: {rmse:.6f}")
    print(f"Final Validation MAE: {mae:.6f}")
    
    print(f"\nPrediction statistics (non-zero only):")
    non_zero_preds = y_pred_masked[non_zero_mask]
    print(f"  Mean: {non_zero_preds.mean():.4f}")
    print(f"  Std: {non_zero_preds.std():.4f}")
    print(f"  Min: {non_zero_preds.min():.4f}")
    print(f"  Max: {non_zero_preds.max():.4f}")
    print(f"  Count: {non_zero_preds.size}")
    
    metrics = {"mse": mse, "mae": mae, "rmse": rmse}
    
    return wrapper, metrics


def _train_discriminative_finetune(
    model,
    train_loader,
    val_loader,
    device,
    base_learning_rate,
    weight_decay,
    epochs,
    patience,
    model_type="hybrid_fno_kan"
):
    """
    判别式微调（Discriminative Fine-tuning）
    不同层使用不同学习率：底层小学习率，顶层大学习率
    """
    print("\n" + "="*80)
    print("DISCRIMINATIVE FINE-TUNING STRATEGY")
    print("="*80)
    
    criterion = MaskedMSELoss()
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # ==================== 设置判别式学习率优化器 ====================
    print(f"\n[Strategy] Discriminative Fine-tuning (lr_decay={DISCRIMINATIVE_LR_DECAY})")
    param_groups = setup_discriminative_lr(model, base_learning_rate, DISCRIMINATIVE_LR_DECAY)
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    
    print("\n" + "-"*80)
    print("Starting training...")
    print("-"*80)
    
    # ==================== 训练循环 ====================
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            X_batch, y_batch, tg_batch, tgt_batch, mask_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Concatenate mask if needed
            if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                X_batch = torch.cat([X_batch, mask_batch], dim=1)
            elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                X_batch = torch.cat([X_batch, mask_batch], dim=1)
            
            optimizer.zero_grad()
            preds = model(X_batch, mask_batch)
            loss = criterion(preds, y_batch, mask_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_batch, tg_batch, tgt_batch, mask_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                    X_batch = torch.cat([X_batch, mask_batch], dim=1)
                elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                    X_batch = torch.cat([X_batch, mask_batch], dim=1)
                
                preds = model(X_batch, mask_batch)
                loss = criterion(preds, y_batch, mask_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # 打印进度
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Advanced finetuning completed. Best val loss: {best_val_loss:.6f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss


def _train_one_stage(
    model, 
    train_loader, 
    val_loader, 
    device,
    learning_rate,
    weight_decay,
    epochs,
    patience,
    stage_name="Training",
    model_type=None
):
    """
    训练一个阶段（预训练或微调）
    """
    if model_type is None:
        model_type = str(MODEL_TYPE).lower().strip()
    
    criterion = MaskedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-7
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"{stage_name}")
    print(f"{'='*60}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Max epochs: {epochs}")
    print(f"Early stopping patience: {patience}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets, target_gain, target_gain_tilt, masks in train_loader:
            inputs, targets, target_gain, target_gain_tilt, masks = \
                inputs.to(device), targets.to(device), target_gain.to(device), \
                target_gain_tilt.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # 根据模型类型决定是否拼接mask
            if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                inputs_for_model = torch.cat([inputs, masks], dim=1)
            elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                inputs_for_model = torch.cat([inputs, masks], dim=1)
            else:
                inputs_for_model = inputs
                
            outputs = model(inputs_for_model, masks)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * masks.sum().item()
        
        total_train_masks = sum(masks.sum().item() for _, _, _, _, masks in train_loader)
        train_loss /= total_train_masks
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, target_gain, target_gain_tilt, masks in val_loader:
                inputs, targets, target_gain, target_gain_tilt, masks = \
                    inputs.to(device), targets.to(device), target_gain.to(device), \
                    target_gain_tilt.to(device), masks.to(device)
                    
                if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                    inputs_for_model = torch.cat([inputs, masks], dim=1)
                elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                    inputs_for_model = torch.cat([inputs, masks], dim=1)
                else:
                    inputs_for_model = inputs
                    
                outputs = model(inputs_for_model, masks)
                loss = criterion(outputs, targets, masks)
                val_loss += loss.item() * masks.sum().item()
        
        total_val_masks = sum(masks.sum().item() for _, _, _, _, masks in val_loader)
        val_loss /= total_val_masks
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # 打印进度
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"{stage_name} completed. Best val loss: {best_val_loss:.6f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss


def train_model_two_stage(
    cosmos_features, cosmos_labels,
    kaggle_features, kaggle_labels,
    test_features,
    preprocessor,
    mask_cols
):
    """
    两阶段训练：
    1. 在 COSMOS 数据集上预训练
    2. 在 Kaggle 数据集上微调
    """
    print("\n" + "="*80)
    print("TWO-STAGE TRAINING: PRETRAIN (COSMOS) + FINETUNE (KAGGLE)")
    print("="*80)
    
    device = torch.device(str(DEVICE)) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # ==================== 准备 COSMOS 数据 ====================
    print("\n[Stage 1] Preparing COSMOS dataset for pretraining...")
    
    # COSMOS 特征预处理
    cosmos_target_gain = cosmos_features['target_gain'].values
    cosmos_target_gain_tilt = cosmos_features['target_gain_tilt'].values
    cosmos_masks = cosmos_features[mask_cols].values
    
    # 使用preprocessor转换cosmos特征（保留target_gain和target_gain_tilt，它们是数值特征）
    cosmos_feature_cols = [c for c in cosmos_features.columns if c not in mask_cols]
    X_cosmos = preprocessor.transform(cosmos_features[cosmos_feature_cols])
    
    # COSMOS 标签
    target_cols = [c for c in cosmos_labels.columns if 'calculated_gain_spectra_' in c]
    target_cols.sort()
    y_cosmos = cosmos_labels[target_cols].values
    
    # 计算baseline和offset
    cosmos_baseline = np.array([
        compute_baseline_gain(tg, tgt) 
        for tg, tgt in zip(cosmos_target_gain, cosmos_target_gain_tilt)
    ])
    cosmos_offset = (y_cosmos - cosmos_baseline) * cosmos_masks
    
    print(f"COSMOS dataset: {len(X_cosmos)} samples")
    
    # 划分训练集和验证集
    X_cosmos_tr, X_cosmos_val, y_cosmos_tr, y_cosmos_val, \
    mask_cosmos_tr, mask_cosmos_val, tg_cosmos_tr, tg_cosmos_val, \
    tgt_cosmos_tr, tgt_cosmos_val = train_test_split(
        X_cosmos, cosmos_offset, cosmos_masks,
        cosmos_target_gain, cosmos_target_gain_tilt,
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    cosmos_train_dataset = OFCDataset(X_cosmos_tr, y_cosmos_tr, tg_cosmos_tr, tgt_cosmos_tr, mask_cosmos_tr)
    cosmos_val_dataset = OFCDataset(X_cosmos_val, y_cosmos_val, tg_cosmos_val, tgt_cosmos_val, mask_cosmos_val)
    
    cosmos_train_loader = DataLoader(cosmos_train_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True)
    cosmos_val_loader = DataLoader(cosmos_val_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=False)
    
    # ==================== 准备 Kaggle 数据 ====================
    print("\n[Stage 2] Preparing Kaggle dataset for finetuning...")
    
    kaggle_target_gain = kaggle_features['target_gain'].values
    kaggle_target_gain_tilt = kaggle_features['target_gain_tilt'].values
    kaggle_masks = kaggle_features[mask_cols].values
    
    # 使用preprocessor转换kaggle特征（保留target_gain和target_gain_tilt，它们是数值特征）
    kaggle_feature_cols = [c for c in kaggle_features.columns if c not in mask_cols]
    X_kaggle = preprocessor.transform(kaggle_features[kaggle_feature_cols])
    
    y_kaggle = kaggle_labels[target_cols].values
    
    kaggle_baseline = np.array([
        compute_baseline_gain(tg, tgt) 
        for tg, tgt in zip(kaggle_target_gain, kaggle_target_gain_tilt)
    ])
    kaggle_offset = (y_kaggle - kaggle_baseline) * kaggle_masks
    
    print(f"Kaggle dataset: {len(X_kaggle)} samples")
    
    X_kaggle_tr, X_kaggle_val, y_kaggle_tr, y_kaggle_val, \
    mask_kaggle_tr, mask_kaggle_val, tg_kaggle_tr, tg_kaggle_val, \
    tgt_kaggle_tr, tgt_kaggle_val = train_test_split(
        X_kaggle, kaggle_offset, kaggle_masks,
        kaggle_target_gain, kaggle_target_gain_tilt,
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    kaggle_train_dataset = OFCDataset(X_kaggle_tr, y_kaggle_tr, tg_kaggle_tr, tgt_kaggle_tr, mask_kaggle_tr)
    kaggle_val_dataset = OFCDataset(X_kaggle_val, y_kaggle_val, tg_kaggle_val, tgt_kaggle_val, mask_kaggle_val)
    
    kaggle_train_loader = DataLoader(kaggle_train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    kaggle_val_loader = DataLoader(kaggle_val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)
    
    # ==================== 创建模型 ====================
    input_dim = X_cosmos.shape[1]
    output_dim = y_cosmos.shape[1]
    
    model_type = str(MODEL_TYPE).lower().strip()
    
    if model_type == "mlp":
        model = SimpleGainPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT
        ).to(device)
    elif model_type in {"fourier_kan", "simple_kan"}:
        if FOURIER_KAN_CONCAT_MASK_INPUT:
            input_dim = input_dim + cosmos_masks.shape[1]
            print(f"[fourier_kan] Concatenating mask into input: model_input_dim={input_dim}")
        model = FourierKANGainPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=FOURIER_KAN_HIDDEN_DIMS,
            dropout=FOURIER_KAN_DROPOUT,
            use_residual=True,
            n_frequencies=FOURIER_KAN_N_FREQUENCIES,
        ).to(device)
    elif model_type == "hybrid_fno_kan":
        if HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
            input_dim = input_dim + cosmos_masks.shape[1]
            print(f"[hybrid_fno_kan] Concatenating mask into input: model_input_dim={input_dim}")
        model = HybridFNOKANPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=HYBRID_FNO_KAN_HIDDEN_DIMS,
            dropout=HYBRID_FNO_KAN_DROPOUT,
            use_residual=True,
            n_frequencies=HYBRID_FNO_KAN_N_FREQUENCIES,
            n_spectral_modes=HYBRID_FNO_KAN_N_SPECTRAL_MODES,
            use_spectral_mixing=HYBRID_FNO_KAN_USE_SPECTRAL_MIXING,
        ).to(device)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE={MODEL_TYPE!r}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # ==================== Stage 1: 预训练 ====================
    pretrain_loss = None
    
    # 检查是否需要加载已有的预训练模型
    if LOAD_PRETRAINED_MODEL and PRETRAIN_MODEL_PATH.exists():
        print("\n" + "="*80)
        print(f"LOADING PRETRAINED MODEL FROM: {PRETRAIN_MODEL_PATH}")
        print("="*80)
        
        try:
            checkpoint = torch.load(PRETRAIN_MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            pretrain_loss = checkpoint.get('pretrain_loss', None)
            
            print(f"✓ Successfully loaded pretrained model!")
            if pretrain_loss:
                print(f"  Previous pretrain validation loss: {pretrain_loss:.6f}")
            print(f"  Model type: {checkpoint.get('model_type', 'unknown')}")
            print(f"  Input dim: {checkpoint.get('input_dim', 'unknown')}")
            print(f"  Output dim: {checkpoint.get('output_dim', 'unknown')}")
            print("\nSkipping Stage 1 (pretraining) and going directly to Stage 2 (finetuning)...")
        
        except Exception as e:
            print(f"✗ Failed to load pretrained model: {e}")
            print("Will perform full two-stage training from scratch...")
            
            # 执行预训练
            model, pretrain_loss = _train_one_stage(
                model=model,
                train_loader=cosmos_train_loader,
                val_loader=cosmos_val_loader,
                device=device,
                learning_rate=PRETRAIN_LEARNING_RATE,
                weight_decay=PRETRAIN_WEIGHT_DECAY,
                epochs=PRETRAIN_EPOCHS,
                patience=PRETRAIN_EARLY_STOPPING_PATIENCE,
                stage_name="STAGE 1: PRETRAINING ON COSMOS",
                model_type=model_type
            )
            
            # 保存预训练模型
            PRETRAIN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'pretrain_loss': pretrain_loss,
            }, PRETRAIN_MODEL_PATH)
            print(f"\nPretrained model saved to: {PRETRAIN_MODEL_PATH}")
    
    else:
        # 不加载预训练模型，从头开始预训练
        if LOAD_PRETRAINED_MODEL:
            print(f"\n⚠ Pretrained model not found at: {PRETRAIN_MODEL_PATH}")
            print("Will perform full two-stage training from scratch...")
        
        model, pretrain_loss = _train_one_stage(
            model=model,
            train_loader=cosmos_train_loader,
            val_loader=cosmos_val_loader,
            device=device,
            learning_rate=PRETRAIN_LEARNING_RATE,
            weight_decay=PRETRAIN_WEIGHT_DECAY,
            epochs=PRETRAIN_EPOCHS,
            patience=PRETRAIN_EARLY_STOPPING_PATIENCE,
            stage_name="STAGE 1: PRETRAINING ON COSMOS",
            model_type=model_type
        )
        
        # 保存预训练模型
        PRETRAIN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'pretrain_loss': pretrain_loss,
        }, PRETRAIN_MODEL_PATH)
        print(f"\nPretrained model saved to: {PRETRAIN_MODEL_PATH}")
    
    # ==================== Stage 2: 判别式微调 ====================
    model, finetune_loss = _train_discriminative_finetune(
        model=model,
        train_loader=kaggle_train_loader,
        val_loader=kaggle_val_loader,
        device=device,
        base_learning_rate=FINETUNE_LEARNING_RATE,
        weight_decay=FINETUNE_WEIGHT_DECAY,
        epochs=FINETUNE_EPOCHS,
        patience=FINETUNE_EARLY_STOPPING_PATIENCE,
        model_type=model_type
    )
    
    # ==================== 最终评估 ====================
    print("\n" + "="*80)
    print("FINAL EVALUATION ON KAGGLE VALIDATION SET")
    print("="*80)
    
    wrapper = PyTorchModelWrapper(model, device)
    y_pred = wrapper.predict(X_kaggle_val, tg_kaggle_val, tgt_kaggle_val, mask_kaggle_val)
    
    y_pred_masked = y_pred * mask_kaggle_val
    
    y_val_denorm = y_kaggle_val + np.array([
        compute_baseline_gain(tg, tgt) 
        for tg, tgt in zip(tg_kaggle_val, tgt_kaggle_val)
    ])
    y_val_masked = y_val_denorm * mask_kaggle_val
    
    non_zero_mask = mask_kaggle_val > 0
    mse = mean_squared_error(y_val_masked[non_zero_mask], y_pred_masked[non_zero_mask])
    mae = mean_absolute_error(y_val_masked[non_zero_mask], y_pred_masked[non_zero_mask])
    rmse = np.sqrt(mse)
    
    print(f"Final Validation MSE: {mse:.6f}")
    print(f"Final Validation RMSE: {rmse:.6f}")
    print(f"Final Validation MAE: {mae:.6f}")
    
    print(f"\nPrediction statistics (non-zero only):")
    non_zero_preds = y_pred_masked[non_zero_mask]
    print(f"  Mean: {non_zero_preds.mean():.4f}")
    print(f"  Std: {non_zero_preds.std():.4f}")
    print(f"  Min: {non_zero_preds.min():.4f}")
    print(f"  Max: {non_zero_preds.max():.4f}")
    print(f"  Count: {non_zero_preds.size}")
    
    metrics = {
        "mse": mse, 
        "mae": mae, 
        "rmse": rmse,
        "pretrain_loss": pretrain_loss,
        "finetune_loss": finetune_loss
    }
    
    return wrapper, metrics
