import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from datetime import datetime
import wandb

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
    HYBRID_FNO_KAN_SPECTRAL_FREQ_RATIO,
    HYBRID_FNO_KAN_USE_SPECTRAL_MIXING,
    HYBRID_FNO_KAN_CONCAT_MASK_INPUT,
    MASK_STRATEGY,
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
    RESNET_MLP_DROPOUT,
    RESNET_MLP_HIDDEN_DIMS,
    FREEZE_LAYERS,
    UNFREEZE_LAST_N_LAYERS,
)

from .network import SimpleGainPredictor, FourierKANGainPredictor, HybridFNOKANPredictor, ResNetPredictor, OFCDataset, TargetNormalizer, compute_baseline_gain

def apply_mask_to_features(X, features_df, mask_cols):
    """
    Apply mask to spectral features in transformed feature array.
    
    Args:
        X: Transformed feature array (numpy array)
        features_df: Original features dataframe (to extract masks)
        mask_cols: List of mask column names
    
    Returns:
        X with mask applied to spectral features
    """
    print("Applying mask to spectral features...")
    
    # Get masks from original dataframe
    masks = features_df[mask_cols].values
    
    # Apply mask to spectral features (first 95 features are spectral)
    spectra_end_idx = len(mask_cols)
    X[:, :spectra_end_idx] = X[:, :spectra_end_idx] * masks
    
    print(f"  Mask applied to {spectra_end_idx} spectral features")
    
    return X

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
            
            mask_strategy = str(MASK_STRATEGY).lower().strip()
            
            if mask is not None:
                tensor_mask = torch.FloatTensor(mask).to(self.device)
                
                if mask_strategy == "concat":
                    # Original approach: concatenate mask as additional input dimension
                    model_type = str(MODEL_TYPE).lower().strip()
                    if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                        tensor_X = torch.cat([tensor_X, tensor_mask], dim=1)
                    elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                        tensor_X = torch.cat([tensor_X, tensor_mask], dim=1)
                    preds_offset = self.model(tensor_X, tensor_mask)
                else:
                    # New approach: mask already applied to features, just pass through
                    preds_offset = self.model(tensor_X)
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


def freeze_layers(model, unfreeze_last_n=2):
    """
    冻结模型大部分层，只解冻最后N层
    """
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
        
    # 获取层组
    layer_groups = get_layer_groups(model)
    num_groups = len(layer_groups)
    
    # 计算需要解冻的层索引
    # 假设layer_groups是从底层到顶层排列的
    # 我们要解冻最后unfreeze_last_n组
    start_unfreeze_idx = max(0, num_groups - unfreeze_last_n)
    
    print(f"Freezing layers: Total groups={num_groups}, Unfreezing last {unfreeze_last_n} groups (indices {start_unfreeze_idx} to {num_groups-1})")
    
    for i in range(start_unfreeze_idx, num_groups):
        print(f"  Unfreezing group {i}")
        for param in layer_groups[i]:
            param.requires_grad = True
            
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")

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
    
    mask_strategy = str(MASK_STRATEGY).lower().strip()
    print(f"Mask strategy: {mask_strategy}")
    
    criterion = MaskedMSELoss()
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # ==================== Layer Freezing ====================
    if FREEZE_LAYERS:
        print("\n[Strategy] Layer Freezing Enabled")
        freeze_layers(model, unfreeze_last_n=UNFREEZE_LAST_N_LAYERS)
    
    # ==================== 设置判别式学习率优化器 ====================
    print(f"\n[Strategy] Discriminative Fine-tuning (lr_decay={DISCRIMINATIVE_LR_DECAY})")
    
    # 只有requires_grad=True的参数会被加入optimizer
    # 我们仍然可以使用setup_discriminative_lr，它会处理层组
    # 但我们需要确保只传递requires_grad=True的参数
    
    # 重新获取层组（包含所有参数）
    layer_groups = get_layer_groups(model)
    num_groups = len(layer_groups)
    
    param_groups = []
    base_lr = base_learning_rate
    
    print("Optimizer groups:")
    for i, params in enumerate(layer_groups):
        # 过滤出需要梯度的参数
        trainable_params_in_group = [p for p in params if p.requires_grad]
        
        if len(trainable_params_in_group) > 0:
            lr_multiplier = DISCRIMINATIVE_LR_DECAY ** (num_groups - 1 - i)
            layer_lr = base_lr * lr_multiplier
            param_groups.append({
                'params': trainable_params_in_group,
                'lr': layer_lr
            })
            print(f"  Group {i} (Active): lr = {layer_lr:.6f}")
        else:
            # print(f"  Group {i} (Frozen): Skipped")
            pass
            
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # Initialize wandb for this stage (if not already active, but typically handled by parent caller or re-init)
    # Note: In two-stage training, we might want to log continuously.
    # For now, let's assume we log with a prefix or as continuation.
    
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
            if mask_strategy == "concat":
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
                
                if mask_strategy == "concat":
                    if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                        X_batch = torch.cat([X_batch, mask_batch], dim=1)
                    elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                        X_batch = torch.cat([X_batch, mask_batch], dim=1)
                
                preds = model(X_batch, mask_batch)
                loss = criterion(preds, y_batch, mask_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_finetune": train_loss,
            "val_loss_finetune": val_loss,
            "learning_rate_finetune": current_lr
        })
        
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
    
    mask_strategy = str(MASK_STRATEGY).lower().strip()
    
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
    print(f"Mask strategy: {mask_strategy}")
    
    # 记录训练开始时间
    training_start_time = time.time()
    epoch_start_time = time.time()
    
    for epoch in tqdm(range(epochs), desc=f"{stage_name} Progress", unit="epoch"):
        # Training
        model.train()
        train_loss = 0.0
        
        # 训练循环添加进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, targets, target_gain, target_gain_tilt, masks in train_pbar:
            inputs, targets, target_gain, target_gain_tilt, masks = \
                inputs.to(device), targets.to(device), target_gain.to(device), \
                target_gain_tilt.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # 根据模型类型决定是否拼接mask
            if mask_strategy == "concat":
                if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                    inputs_for_model = torch.cat([inputs, masks], dim=1)
                elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                    inputs_for_model = torch.cat([inputs, masks], dim=1)
                else:
                    inputs_for_model = inputs
            else:
                inputs_for_model = inputs
                
            outputs = model(inputs_for_model, masks)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * masks.sum().item()
            
            # 更新进度条显示当前loss
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        total_train_masks = sum(masks.sum().item() for _, _, _, _, masks in train_loader)
        train_loss /= total_train_masks
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # 验证循环添加进度条
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for inputs, targets, target_gain, target_gain_tilt, masks in val_pbar:
                inputs, targets, target_gain, target_gain_tilt, masks = \
                    inputs.to(device), targets.to(device), target_gain.to(device), \
                    target_gain_tilt.to(device), masks.to(device)
                    
                if mask_strategy == "concat":
                    if model_type in {"fourier_kan", "simple_kan"} and FOURIER_KAN_CONCAT_MASK_INPUT:
                        inputs_for_model = torch.cat([inputs, masks], dim=1)
                    elif model_type == "hybrid_fno_kan" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
                        inputs_for_model = torch.cat([inputs, masks], dim=1)
                    else:
                        inputs_for_model = inputs
                else:
                    inputs_for_model = inputs
                    
                outputs = model(inputs_for_model, masks)
                loss = criterion(outputs, targets, masks)
                val_loss += loss.item() * masks.sum().item()
                
                # 更新进度条显示当前loss
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        total_val_masks = sum(masks.sum().item() for _, _, _, _, masks in val_loader)
        val_loss /= total_val_masks
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            f"train_loss_{stage_name.replace(' ', '_').lower()}": train_loss,
            f"val_loss_{stage_name.replace(' ', '_').lower()}": val_loss,
            f"learning_rate_{stage_name.replace(' ', '_').lower()}": current_lr
        })
        
        # 打印进度
        if (epoch + 1) % 20 == 0:
            # 计算从训练开始到现在的总时间
            total_elapsed = time.time() - training_start_time
            # 计算最近20个epoch的时间
            epoch_elapsed = time.time() - epoch_start_time
            epoch_start_time = time.time()  # 重置计时器
            
            # 格式化时间显示
            total_hours, total_remainder = divmod(int(total_elapsed), 3600)
            total_minutes, total_seconds = divmod(total_remainder, 60)
            epoch_minutes, epoch_seconds = divmod(int(epoch_elapsed), 60)
            
            print(f"\nEpoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
            print(f"  └─ Last 20 epochs: {epoch_minutes}m {epoch_seconds}s | "
                  f"Total time: {total_hours}h {total_minutes}m {total_seconds}s\n")
        
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
    
    # 计算总训练时间
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(int(total_training_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{stage_name} completed!")
    print(f"  └─ Best val loss: {best_val_loss:.6f}")
    print(f"  └─ Total training time: {hours}h {minutes}m {seconds}s\n")
    
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
    
    if torch.cuda.is_available():
        device = torch.device(str(DEVICE))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 检查mask策略
    mask_strategy = str(MASK_STRATEGY).lower().strip()
    print(f"\nMask strategy: {mask_strategy}")
    apply_mask_to_spectra = (mask_strategy == "multiply")
    
    # ==================== 准备 COSMOS 数据 ====================
    print("\n[Stage 1] Preparing COSMOS dataset for pretraining...")
    
    # COSMOS 特征预处理
    cosmos_target_gain = cosmos_features['target_gain'].values
    cosmos_target_gain_tilt = cosmos_features['target_gain_tilt'].values
    cosmos_masks = cosmos_features[mask_cols].values
    
    # 使用preprocessor转换cosmos特征（保留target_gain和target_gain_tilt，它们是数值特征）
    cosmos_feature_cols = [c for c in cosmos_features.columns if c not in mask_cols]
    X_cosmos = preprocessor.transform(cosmos_features[cosmos_feature_cols])
    
    # 如果使用multiply策略，在转换后应用mask
    if apply_mask_to_spectra:
        X_cosmos = apply_mask_to_features(X_cosmos, cosmos_features, mask_cols)
    
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
    
    loader_kwargs = {'pin_memory': True} if device.type != 'cpu' else {}
    
    cosmos_train_loader = DataLoader(cosmos_train_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True, **loader_kwargs)
    cosmos_val_loader = DataLoader(cosmos_val_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=False, **loader_kwargs)
    
    # ==================== 准备 Kaggle 数据 ====================
    print("\n[Stage 2] Preparing Kaggle dataset for finetuning...")
    
    kaggle_target_gain = kaggle_features['target_gain'].values
    kaggle_target_gain_tilt = kaggle_features['target_gain_tilt'].values
    kaggle_masks = kaggle_features[mask_cols].values
    
    # 使用preprocessor转换kaggle特征（保留target_gain和target_gain_tilt，它们是数值特征）
    kaggle_feature_cols = [c for c in kaggle_features.columns if c not in mask_cols]
    X_kaggle = preprocessor.transform(kaggle_features[kaggle_feature_cols])
    
    # 如果使用multiply策略，在转换后应用mask
    if apply_mask_to_spectra:
        X_kaggle = apply_mask_to_features(X_kaggle, kaggle_features, mask_cols)
    
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
    
    loader_kwargs = {'pin_memory': True} if device.type != 'cpu' else {}
    
    kaggle_train_loader = DataLoader(kaggle_train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, **loader_kwargs)
    kaggle_val_loader = DataLoader(kaggle_val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, **loader_kwargs)
    
    # ==================== 创建模型 ====================
    input_dim = X_cosmos.shape[1]
    output_dim = y_cosmos.shape[1]
    
    mask_strategy = str(MASK_STRATEGY).lower().strip()
    print(f"\nMask strategy: {mask_strategy}")
    
    model_type = str(MODEL_TYPE).lower().strip()
    
    if model_type == "mlp":
        model = SimpleGainPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT
        ).to(device)
    elif model_type in {"fourier_kan", "simple_kan"}:
        if mask_strategy == "concat" and FOURIER_KAN_CONCAT_MASK_INPUT:
            input_dim = input_dim + cosmos_masks.shape[1]
            print(f"[fourier_kan] Concatenating mask into input: model_input_dim={input_dim}")
        else:
            print(f"[fourier_kan] Using mask strategy: {mask_strategy} (no concatenation)")
        model = FourierKANGainPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=FOURIER_KAN_HIDDEN_DIMS,
            dropout=FOURIER_KAN_DROPOUT,
            use_residual=True,
            n_frequencies=FOURIER_KAN_N_FREQUENCIES,
        ).to(device)
    elif model_type == "hybrid_fno_kan":
        if mask_strategy == "concat" and HYBRID_FNO_KAN_CONCAT_MASK_INPUT:
            input_dim = input_dim + cosmos_masks.shape[1]
            print(f"[hybrid_fno_kan] Concatenating mask into input: model_input_dim={input_dim}")
        else:
            print(f"[hybrid_fno_kan] Using mask strategy: {mask_strategy} (no concatenation)")
        model = HybridFNOKANPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=HYBRID_FNO_KAN_HIDDEN_DIMS,
            dropout=HYBRID_FNO_KAN_DROPOUT,
            use_residual=True,
            n_frequencies=HYBRID_FNO_KAN_N_FREQUENCIES,
            spectral_freq_ratio=HYBRID_FNO_KAN_SPECTRAL_FREQ_RATIO,
            use_spectral_mixing=HYBRID_FNO_KAN_USE_SPECTRAL_MIXING,
        ).to(device)
    elif model_type == "resnet_mlp":
        if mask_strategy == "concat":
            input_dim = input_dim + cosmos_masks.shape[1]
            print(f"[resnet_mlp] Concatenating mask into input: model_input_dim={input_dim}")
        
        model = ResNetPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=RESNET_MLP_HIDDEN_DIMS,
            dropout=RESNET_MLP_DROPOUT
        ).to(device)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE={MODEL_TYPE!r}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        config={
            "model_type": MODEL_TYPE,
            "mask_strategy": MASK_STRATEGY,
            "pretrain_learning_rate": PRETRAIN_LEARNING_RATE,
            "finetune_learning_rate": FINETUNE_LEARNING_RATE,
            "weight_decay": PRETRAIN_WEIGHT_DECAY,
            "pretrain_batch_size": PRETRAIN_BATCH_SIZE,
            "finetune_batch_size": FINETUNE_BATCH_SIZE,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "dropout": DROPOUT,
            "hidden_dims": HIDDEN_DIMS,
            "device": str(device),
            "stage": "two_stage"
        }
    )
    
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
    
    # Log final metrics to wandb
    wandb.log(metrics)
    wandb.finish()
    
    return wrapper, metrics
