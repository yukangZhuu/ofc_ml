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
    """
    Wrapper to make PyTorch model behave like sklearn estimator for consistency.
    """
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

def train_model(X_train, y_train):
    """
    Train a Neural Network model.
    
    Returns:
        model: PyTorchModelWrapper.
        metrics: Dictionary of validation metrics.
    """
    print("Training Neural Network model...")
    
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Create Datasets and Loaders
    train_dataset = OFCDataset(X_tr, y_tr)
    val_dataset = OFCDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize Model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = GainPredictor(input_dim, output_dim).to(device)
    
    # Training Config
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # verbose deprecated in recent torch versions or behavior changed, removing it for safety
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    epochs = 200
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_dataset)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    # Final Validation Metrics
    wrapper = PyTorchModelWrapper(model, device)
    y_pred = wrapper.predict(X_val)
    
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"Final Validation MSE: {mse:.4f}")
    print(f"Final Validation MAE: {mae:.4f}")
    
    metrics = {"mse": mse, "mae": mae}
    return wrapper, metrics
