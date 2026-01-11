import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from .config import RANDOM_STATE, TEST_SIZE
from .network import GainPredictor, OFCDataset

class PyTorchModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).to(self.device)
            preds = self.model(tensor_X)
            return preds.cpu().numpy()

def train_model(X_train, y_train, preprocessor):
    print("Training CNN-based Neural Network model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    train_dataset = OFCDataset(X_tr, y_tr)
    val_dataset = OFCDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    num_spectral = 95
    num_scalar = 4
    num_mask = 95
    
    cat_transformer = preprocessor.named_transformers_['cat']
    if cat_transformer is not None:
        num_cat = cat_transformer.get_feature_names_out().shape[0]
    else:
        num_cat = 0
    
    print(f"Model architecture:")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Output dimension: {output_dim}")
    print(f"  - Scalar features: {num_scalar}")
    print(f"  - Spectral features: {num_spectral}")
    print(f"  - Categorical features (after one-hot): {num_cat}")
    print(f"  - Mask features: {num_mask}")
    
    model = GainPredictor(
        input_dim=input_dim, 
        output_dim=output_dim,
        num_spectral=num_spectral,
        num_scalar=num_scalar,
        num_cat=num_cat,
        num_mask=num_mask
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    epochs = 300
    best_val_loss = float('inf')
    best_model_state = None
    patience = 30
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_dataset)
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
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
    y_pred = wrapper.predict(X_val)
    
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Final Validation MSE: {mse:.6f}")
    print(f"Final Validation RMSE: {rmse:.6f}")
    print(f"Final Validation MAE: {mae:.6f}")
    
    metrics = {"mse": mse, "mae": mae, "rmse": rmse}
    return wrapper, metrics
