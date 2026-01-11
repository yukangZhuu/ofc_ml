import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import wandb

from .config import RANDOM_STATE, TEST_SIZE, WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE
from .network import SimpleGainPredictor, OFCDataset, TargetNormalizer

class PyTorchModelWrapper:
    def __init__(self, model, target_normalizer, device):
        self.model = model
        self.target_normalizer = target_normalizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).to(self.device)
            preds = self.model(tensor_X)
            preds = preds.cpu().numpy()
            preds = self.target_normalizer.inverse_transform(preds)
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

class OFCDatasetWithMask(Dataset):
    def __init__(self, X, y, masks):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.masks = torch.FloatTensor(masks)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx]

def train_model(X_train, y_train, preprocessor, train_features, mask_cols):
    print("Training simplified Neural Network model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    target_normalizer = TargetNormalizer()
    y_train_normalized = target_normalizer.fit_transform(y_train)
    
    print(f"Target normalization:")
    print(f"  Mean: {target_normalizer.mean.mean():.4f}")
    print(f"  Std: {target_normalizer.std.mean():.4f}")
    
    train_masks = train_features[mask_cols].values
    
    X_tr, X_val, y_tr, y_val, mask_tr, mask_val = train_test_split(
        X_train, y_train_normalized, train_masks, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    train_dataset = OFCDatasetWithMask(X_tr, y_tr, mask_tr)
    val_dataset = OFCDatasetWithMask(X_val, y_val, mask_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = SimpleGainPredictor(
        input_dim=input_dim, 
        output_dim=output_dim,
        hidden_dims=[256, 128],
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 200,
        'optimizer': 'Adam',
        'weight_decay': 1e-5,
        'scheduler': 'ReduceLROnPlateau',
        'patience': 20,
        'factor': 0.5,
        'min_lr': 1e-6,
        'early_stopping_patience': 50,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': [256, 128],
        'dropout': 0.3,
        'device': str(device),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'random_state': RANDOM_STATE,
        'total_params': total_params
    }
    
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        config=config,
        name='simple-gain-predictor',
        tags=['mlp', 'normalized', 'edfa']
    )
    
    wandb.watch(model, log_freq=100, log_graph=True)
    
    criterion = MaskedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )
    
    epochs = 200
    best_val_loss = float('inf')
    best_model_state = None
    patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets, masks in train_loader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * masks.sum().item()
            
        total_train_masks = sum(masks.sum().item() for _, _, masks in train_loader)
        train_loss /= total_train_masks
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)
                val_loss += loss.item() * masks.sum().item()
        
        total_val_masks = sum(masks.sum().item() for _, _, masks in val_loader)
        val_loss /= total_val_masks
        
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr
        })
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.6f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            wandb.run.summary['best_val_loss'] = best_val_loss
            wandb.run.summary['best_epoch'] = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    wrapper = PyTorchModelWrapper(model, target_normalizer, device)
    y_pred = wrapper.predict(X_val)
    
    y_pred_masked = y_pred * mask_val
    y_val_denorm = target_normalizer.inverse_transform(y_val)
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
    
    wandb.log({
        'final_mse': mse,
        'final_rmse': rmse,
        'final_mae': mae
    })
    
    wandb.run.summary['final_val_loss'] = best_val_loss
    wandb.run.summary['final_mse'] = mse
    wandb.run.summary['final_rmse'] = rmse
    wandb.run.summary['final_mae'] = mae
    
    metrics = {"mse": mse, "mae": mae, "rmse": rmse}
    
    wandb.finish()
    
    return wrapper, metrics