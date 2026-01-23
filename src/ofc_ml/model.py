import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
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
    PRETRAIN_VAL_EVERY_N_EPOCHS,
    FINETUNE_LEARNING_RATE,
    FINETUNE_WEIGHT_DECAY,
    FINETUNE_BATCH_SIZE,
    FINETUNE_EPOCHS,
    FINETUNE_EARLY_STOPPING_PATIENCE,
    PRETRAIN_MODEL_PATH,
    USE_MIXED_PRECISION,
)
from .network import HybridFNOKANPredictor, OFCDataset, compute_baseline_gain


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

            baseline = compute_baseline_gain(target_gain, target_gain_tilt)

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


def prepare_data(features, labels, preprocessor, mask_cols, target_cols=None):
    """Extract and process features and targets from raw dataframes."""
    target_gain = features['target_gain'].values
    target_gain_tilt = features['target_gain_tilt'].values
    masks = features[mask_cols].values

    feature_cols = [c for c in features.columns if c not in mask_cols]
    
    # Check if we are in 'multiply' mode, which needs mask columns
    # We can infer this by checking if the preprocessor expects mask columns
    # A simple way is to pass the entire dataframe if feature_cols doesn't match preprocessor expectations
    # But a cleaner way is to just pass the whole dataframe, as ColumnTransformer ignores extra columns if remainder='drop'
    X = preprocessor.transform(features)

    y_offset = None
    if labels is not None and target_cols is not None:
        y = labels[target_cols].values
        baseline = compute_baseline_gain(target_gain, target_gain_tilt)
        y_offset = (y - baseline) * masks

    return X, y_offset, target_gain, target_gain_tilt, masks


def create_dataloaders(X, y_offset, tg, tgt, masks, batch_size, test_size, random_state, device):
    """Split data and create PyTorch DataLoaders."""
    X_tr, X_val, y_tr, y_val, mask_tr, mask_val, tg_tr, tg_val, tgt_tr, tgt_val = train_test_split(
        X, y_offset, masks, tg, tgt, test_size=test_size, random_state=random_state
    )

    train_dataset = OFCDataset(X_tr, y_tr, tg_tr, tgt_tr, mask_tr)
    val_dataset = OFCDataset(X_val, y_val, tg_val, tgt_val, mask_val)

    num_workers = 4 if device.type == 'cuda' else 0
    pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # Return validation components for final evaluation
    val_data = (X_val, y_val, tg_val, tgt_val, mask_val)
    return train_loader, val_loader, val_data


class Trainer:
    """Handles the training loop, validation, and early stopping."""
    def __init__(self, model, device, criterion=None, use_amp=False):
        self.model = model
        self.device = device
        self.criterion = criterion or MaskedMSELoss()
        self.scaler = GradScaler() if use_amp and device.type == 'cuda' else None
        
        if self.scaler:
            print("Mixed precision training: Enabled (FP16)")

    def train_epoch(self, loader, optimizer):
        self.model.train()
        total_loss = 0.0
        total_masks = 0.0
        
        pbar = tqdm(loader, leave=False, desc="Training")
        for batch in pbar:
            X, y, _, _, mask = [b.to(self.device) for b in batch]
            optimizer.zero_grad()
            
            with autocast(device_type=self.device.type, enabled=self.scaler is not None):
                preds = self.model(X, mask)
                loss = self.criterion(preds, y, mask)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
            loss_val = loss.item()
            mask_sum = mask.sum().item()
            total_loss += loss_val * mask_sum
            total_masks += mask_sum
            
            pbar.set_postfix({'loss': f'{loss_val:.6f}'})
            
        return total_loss / total_masks if total_masks > 0 else 0.0

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_masks = 0.0
        
        with torch.no_grad():
            for batch in loader:
                X, y, _, _, mask = [b.to(self.device) for b in batch]
                
                with autocast(device_type=self.device.type, enabled=self.scaler is not None):
                    preds = self.model(X, mask)
                    loss = self.criterion(preds, y, mask)
                    
                total_loss += loss.item() * mask.sum().item()
                total_masks += mask.sum().item()
                
        return total_loss / total_masks if total_masks > 0 else 0.0

    def fit(self, train_loader, val_loader, optimizer, epochs, patience, 
            scheduler=None, val_every_n=1, title="Training"):
        print(f"\n{'='*60}\n{title}\n{'='*60}")
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            
            if (epoch + 1) % val_every_n == 0:
                val_loss = self.validate(val_loader)
                
                if scheduler:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                        
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - Val: {val_loss:.6f} - LR: {current_lr:.8f}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        
        if best_state:
            self.model.load_state_dict(best_state)
            
        return best_loss


def train_model_two_stage(
    cosmos_features, cosmos_labels,
    kaggle_features, kaggle_labels,
    test_features,
    preprocessor,
    mask_cols
):
    """
    Two-Stage Training:
    1. Pretrain on COSMOS dataset.
    2. Discriminative Finetune on Kaggle dataset.
    """
    device = torch.device(str(DEVICE)) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. Data Preparation
    # -------------------------------------------------------------------------
    target_cols = [c for c in cosmos_labels.columns if 'calculated_gain_spectra_' in c]
    target_cols.sort()

    # COSMOS Data
    print("\n[Data] Preparing COSMOS dataset...")
    X_cosmos, y_offset_cosmos, tg_cosmos, tgt_cosmos, mask_cosmos = prepare_data(
        cosmos_features, cosmos_labels, preprocessor, mask_cols, target_cols
    )
    cosmos_loaders = create_dataloaders(
        X_cosmos, y_offset_cosmos, tg_cosmos, tgt_cosmos, mask_cosmos,
        PRETRAIN_BATCH_SIZE, TEST_SIZE, RANDOM_STATE, device
    )
    cosmos_train_loader, cosmos_val_loader, _ = cosmos_loaders

    # Kaggle Data
    print("[Data] Preparing Kaggle dataset...")
    X_kaggle, y_offset_kaggle, tg_kaggle, tgt_kaggle, mask_kaggle = prepare_data(
        kaggle_features, kaggle_labels, preprocessor, mask_cols, target_cols
    )
    kaggle_loaders = create_dataloaders(
        X_kaggle, y_offset_kaggle, tg_kaggle, tgt_kaggle, mask_kaggle,
        FINETUNE_BATCH_SIZE, TEST_SIZE, RANDOM_STATE, device
    )
    kaggle_train_loader, kaggle_val_loader, kaggle_val_data = kaggle_loaders
    X_k_val, y_offset_k_val, _, _, mask_k_val = kaggle_val_data

    # -------------------------------------------------------------------------
    # 2. Model Initialization
    # -------------------------------------------------------------------------
    model = HybridFNOKANPredictor(
        input_dim=X_cosmos.shape[1],
        output_dim=y_offset_cosmos.shape[1],
        hidden_dims=HYBRID_FNO_KAN_HIDDEN_DIMS,
        dropout=HYBRID_FNO_KAN_DROPOUT,
        use_residual=True,
        n_frequencies=HYBRID_FNO_KAN_N_FREQUENCIES,
        spectral_freq_ratio=HYBRID_FNO_KAN_SPECTRAL_FREQ_RATIO,
        use_spectral_mixing=HYBRID_FNO_KAN_USE_SPECTRAL_MIXING,
    ).to(device)

    print(f"\nModel Params: {sum(p.numel() for p in model.parameters()):,}")
    trainer = Trainer(model, device, use_amp=USE_MIXED_PRECISION)

    # -------------------------------------------------------------------------
    # 3. Stage 1: Pretraining (COSMOS)
    # -------------------------------------------------------------------------
    pretrain_loss = None
    run_pretraining = True

    if LOAD_PRETRAINED_MODEL and PRETRAIN_MODEL_PATH.exists():
        try:
            print(f"\n[Pretrain] Loading model from {PRETRAIN_MODEL_PATH}")
            checkpoint = torch.load(PRETRAIN_MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            pretrain_loss = checkpoint.get('pretrain_loss')
            run_pretraining = False
            print("✓ Pretrained model loaded.")
        except Exception as e:
            print(f"✗ Failed to load model: {e}. Retraining...")

    if run_pretraining:
        optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LEARNING_RATE, weight_decay=PRETRAIN_WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
        
        pretrain_loss = trainer.fit(
            cosmos_train_loader, cosmos_val_loader, optimizer,
            epochs=PRETRAIN_EPOCHS, patience=PRETRAIN_EARLY_STOPPING_PATIENCE,
            scheduler=scheduler, val_every_n=PRETRAIN_VAL_EVERY_N_EPOCHS,
            title="Stage 1: Pretraining (COSMOS)"
        )
        
        # Save Pretrained Model
        PRETRAIN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'pretrain_loss': pretrain_loss,
            'input_dim': X_cosmos.shape[1],
            'output_dim': y_offset_cosmos.shape[1]
        }, PRETRAIN_MODEL_PATH)
        print(f"Pretrained model saved to {PRETRAIN_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 4. Stage 2: Finetuning (Kaggle)
    # -------------------------------------------------------------------------
    print("\n[Finetune] Starting finetuning...")
    optimizer_ft = optim.AdamW(model.parameters(), lr=FINETUNE_LEARNING_RATE, weight_decay=FINETUNE_WEIGHT_DECAY)
    
    finetune_loss = trainer.fit(
        kaggle_train_loader, kaggle_val_loader, optimizer_ft,
        epochs=FINETUNE_EPOCHS, patience=FINETUNE_EARLY_STOPPING_PATIENCE,
        title="Stage 2: Finetuning (Kaggle)"
    )

    # -------------------------------------------------------------------------
    # 5. Final Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    wrapper = PyTorchModelWrapper(model, device)
    
    # Calculate metrics on Kaggle validation set
    y_pred_offset = model(torch.FloatTensor(X_k_val).to(device), torch.FloatTensor(mask_k_val).to(device))
    y_pred_offset = y_pred_offset.detach().cpu().numpy()
    
    y_pred_masked = y_pred_offset * mask_k_val
    y_true_masked = y_offset_k_val * mask_k_val
    
    non_zero = mask_k_val > 0
    mse = mean_squared_error(y_true_masked[non_zero], y_pred_masked[non_zero])
    mae = mean_absolute_error(y_true_masked[non_zero], y_pred_masked[non_zero])
    rmse = np.sqrt(mse)
    
    print(f"Validation MSE: {mse:.6f}")
    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation MAE: {mae:.6f}")

    metrics = {
        "mse": mse, "mae": mae, "rmse": rmse,
        "pretrain_loss": pretrain_loss,
        "finetune_loss": finetune_loss
    }

    return wrapper, metrics
