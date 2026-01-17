import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from pathlib import Path

from .config import RANDOM_STATE, TEST_SIZE, WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, DROPOUT, HIDDEN_DIMS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, EARLY_STOPPING_PATIENCE
from .network import SimpleGainPredictor, OFCDataset, TargetNormalizer, compute_baseline_gain

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

def train_model(X_train, y_train, preprocessor, train_features, mask_cols):
    print("Training simplified Neural Network model with baseline+offset approach...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    model = SimpleGainPredictor(
        input_dim=input_dim, 
        output_dim=output_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    criterion = MaskedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    
    epochs = 500
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
            outputs = model(inputs, masks)
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
                outputs = model(inputs, masks)
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
