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

from .config import (
    RANDOM_STATE,
    TEST_SIZE,
    HYBRID_FNO_KAN_DROPOUT,
    HYBRID_FNO_KAN_HIDDEN_DIMS,
    HYBRID_FNO_KAN_N_FREQUENCIES,
    HYBRID_FNO_KAN_SPECTRAL_FREQ_RATIO,
    HYBRID_FNO_KAN_USE_SPECTRAL_MIXING,
    DEVICE,
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
from .network import HybridFNOKANPredictor, OFCDataset, TargetNormalizer, compute_baseline_gain

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
                preds_offset = self.model(tensor_X, tensor_mask)
            else:
                preds_offset = self.model(tensor_X)

            preds_offset = preds_offset.cpu().numpy()

            baseline = np.array([
                compute_baseline_gain(tg, tgt)
                for tg, tgt in zip(target_gain, target_gain_tilt)
            ])

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

    named_modules = list(model.named_children())

    if len(named_modules) == 0:
        return [list(model.parameters())]

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
    patience
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

    print(f"\n[Strategy] Discriminative Fine-tuning (lr_decay={DISCRIMINATIVE_LR_DECAY})")
    param_groups = setup_discriminative_lr(model, base_learning_rate, DISCRIMINATIVE_LR_DECAY)
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    print("\n" + "-"*80)
    print("Starting training...")
    print("-"*80)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            X_batch, y_batch, tg_batch, tgt_batch, mask_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch, mask_batch)
            loss = criterion(preds, y_batch, mask_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_batch, tg_batch, tgt_batch, mask_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)

                preds = model(X_batch, mask_batch)
                loss = criterion(preds, y_batch, mask_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.6f}")

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
    stage_name="Training"
):
    """
    训练一个阶段（预训练或微调）
    """
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

    training_start_time = time.time()
    epoch_start_time = time.time()
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    for epoch in tqdm(range(epochs), desc=f"{stage_name} Progress", unit="epoch"):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        batch_times = []
        for inputs, targets, target_gain, target_gain_tilt, masks in train_pbar:
            batch_start = time.time()
            
            inputs, targets, target_gain, target_gain_tilt, masks = \
                inputs.to(device), targets.to(device), target_gain.to(device), \
                target_gain_tilt.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(inputs, masks)
            loss = criterion(outputs, targets, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * masks.sum().item()
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            train_pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'batch_time': f'{batch_time*1000:.1f}ms'
            })
        
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        samples_per_sec = len(train_loader.dataset) / (sum(batch_times) if batch_times else 1)
        
        if device.type == 'cuda' and epoch == 0:
            print(f"  First epoch stats:")
            print(f"    Avg batch time: {avg_batch_time*1000:.1f}ms")
            print(f"    Samples/sec: {samples_per_sec:.1f}")
            print(f"    GPU Memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        total_train_masks = sum(masks.sum().item() for _, _, _, _, masks in train_loader)
        train_loss /= total_train_masks

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for inputs, targets, target_gain, target_gain_tilt, masks in val_pbar:
                inputs, targets, target_gain, target_gain_tilt, masks = \
                    inputs.to(device), targets.to(device), target_gain.to(device), \
                    target_gain_tilt.to(device), masks.to(device)

                outputs = model(inputs, masks)
                loss = criterion(outputs, targets, masks)
                val_loss += loss.item() * masks.sum().item()

                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        total_val_masks = sum(masks.sum().item() for _, _, _, _, masks in val_loader)
        val_loss /= total_val_masks

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            total_elapsed = time.time() - training_start_time
            epoch_elapsed = time.time() - epoch_start_time
            epoch_start_time = time.time()

            total_hours, total_remainder = divmod(int(total_elapsed), 3600)
            total_minutes, total_seconds = divmod(total_remainder, 60)
            epoch_minutes, epoch_seconds = divmod(int(epoch_elapsed), 60)

            print(f"\nEpoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
            print(f"  └─ Last 20 epochs: {epoch_minutes}m {epoch_seconds}s | "
                  f"Total time: {total_hours}h {total_minutes}m {total_seconds}s\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(int(total_training_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{stage_name} completed!")
    print(f"  └─ Best val loss: {best_val_loss:.6f}")
    print(f"  └─ Total training time: {hours}h {minutes}m {seconds}s\n")

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
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU Memory Free: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    print("\n[Stage 1] Preparing COSMOS dataset for pretraining...")

    cosmos_target_gain = cosmos_features['target_gain'].values
    cosmos_target_gain_tilt = cosmos_features['target_gain_tilt'].values
    cosmos_masks = cosmos_features[mask_cols].values

    cosmos_feature_cols = [c for c in cosmos_features.columns if c not in mask_cols]
    X_cosmos = preprocessor.transform(cosmos_features[cosmos_feature_cols])

    target_cols = [c for c in cosmos_labels.columns if 'calculated_gain_spectra_' in c]
    target_cols.sort()
    y_cosmos = cosmos_labels[target_cols].values

    cosmos_baseline = np.array([
        compute_baseline_gain(tg, tgt)
        for tg, tgt in zip(cosmos_target_gain, cosmos_target_gain_tilt)
    ])
    cosmos_offset = (y_cosmos - cosmos_baseline) * cosmos_masks

    print(f"COSMOS dataset: {len(X_cosmos)} samples")

    X_cosmos_tr, X_cosmos_val, y_cosmos_tr, y_cosmos_val, \
    mask_cosmos_tr, mask_cosmos_val, tg_cosmos_tr, tg_cosmos_val, \
    tgt_cosmos_tr, tgt_cosmos_val = train_test_split(
        X_cosmos, cosmos_offset, cosmos_masks,
        cosmos_target_gain, cosmos_target_gain_tilt,
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    cosmos_train_dataset = OFCDataset(X_cosmos_tr, y_cosmos_tr, tg_cosmos_tr, tgt_cosmos_tr, mask_cosmos_tr)
    cosmos_val_dataset = OFCDataset(X_cosmos_val, y_cosmos_val, tg_cosmos_val, tgt_cosmos_val, mask_cosmos_val)

    num_workers = 4 if device.type == 'cuda' else 0
    pin_memory = device.type == 'cuda'
    
    cosmos_train_loader = DataLoader(
        cosmos_train_dataset, 
        batch_size=PRETRAIN_BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    cosmos_val_loader = DataLoader(
        cosmos_val_dataset, 
        batch_size=PRETRAIN_BATCH_SIZE, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print("\n[Stage 2] Preparing Kaggle dataset for finetuning...")

    kaggle_target_gain = kaggle_features['target_gain'].values
    kaggle_target_gain_tilt = kaggle_features['target_gain_tilt'].values
    kaggle_masks = kaggle_features[mask_cols].values

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

    kaggle_train_loader = DataLoader(
        kaggle_train_dataset, 
        batch_size=FINETUNE_BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    kaggle_val_loader = DataLoader(
        kaggle_val_dataset, 
        batch_size=FINETUNE_BATCH_SIZE, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    input_dim = X_cosmos.shape[1]
    output_dim = y_cosmos.shape[1]

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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    pretrain_loss = None

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

            model, pretrain_loss = _train_one_stage(
                model=model,
                train_loader=cosmos_train_loader,
                val_loader=cosmos_val_loader,
                device=device,
                learning_rate=PRETRAIN_LEARNING_RATE,
                weight_decay=PRETRAIN_WEIGHT_DECAY,
                epochs=PRETRAIN_EPOCHS,
                patience=PRETRAIN_EARLY_STOPPING_PATIENCE,
                stage_name="STAGE 1: PRETRAINING ON COSMOS"
            )

            PRETRAIN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': 'hybrid_fno_kan',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'pretrain_loss': pretrain_loss,
            }, PRETRAIN_MODEL_PATH)
            print(f"\nPretrained model saved to: {PRETRAIN_MODEL_PATH}")

    else:
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
            stage_name="STAGE 1: PRETRAINING ON COSMOS"
        )

        PRETRAIN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'hybrid_fno_kan',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'pretrain_loss': pretrain_loss,
        }, PRETRAIN_MODEL_PATH)
        print(f"\nPretrained model saved to: {PRETRAIN_MODEL_PATH}")

    model, finetune_loss = _train_discriminative_finetune(
        model=model,
        train_loader=kaggle_train_loader,
        val_loader=kaggle_val_loader,
        device=device,
        base_learning_rate=FINETUNE_LEARNING_RATE,
        weight_decay=FINETUNE_WEIGHT_DECAY,
        epochs=FINETUNE_EPOCHS,
        patience=FINETUNE_EARLY_STOPPING_PATIENCE
    )

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
