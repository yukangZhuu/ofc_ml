import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

def compute_baseline_gain(target_gain, target_gain_tilt, num_channels=95):
    """
    Compute baseline gain for each channel using target_gain and target_gain_tilt.
    
    Args:
        target_gain: scalar or tensor, baseline gain value at center channel
        target_gain_tilt: scalar or tensor, tilt value across channels
        num_channels: int, number of channels (default 95)
    
    Returns:
        baseline: numpy array or tensor of shape (batch_size, num_channels), baseline gain for each channel
    """
    center_idx = (num_channels - 1) / 2  # Center index (47 for 95 channels)
    
    # Create channel indices
    channel_indices = np.arange(num_channels)
    
    # Handle both scalar and batch inputs
    if np.isscalar(target_gain):
        baseline = target_gain + target_gain_tilt * (center_idx - channel_indices) / (num_channels - 1)
    else:
        # Batch processing
        batch_size = len(target_gain)
        baseline = np.zeros((batch_size, num_channels))
        for i in range(batch_size):
            baseline[i] = target_gain[i] + target_gain_tilt[i] * (center_idx - channel_indices) / (num_channels - 1)
    
    return baseline

class OFCDataset(Dataset):
    def __init__(self, X, y, target_gain, target_gain_tilt, mask=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.target_gain = torch.FloatTensor(target_gain)
        self.target_gain_tilt = torch.FloatTensor(target_gain_tilt)
        if mask is not None:
            self.mask = torch.FloatTensor(mask)
        else:
            self.mask = None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.mask is not None:
            return self.X[idx], self.y[idx], self.target_gain[idx], self.target_gain_tilt[idx], self.mask[idx]
        else:
            return self.X[idx], self.y[idx], self.target_gain[idx], self.target_gain_tilt[idx], torch.ones(self.y.shape[1])

class SimpleGainPredictor(nn.Module):
    def __init__(self, input_dim=99, output_dim=95, hidden_dims=[256, 128], dropout=0.3):
        super(SimpleGainPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, target_gain=None, target_gain_tilt=None, mask=None):
        offset_normalized = self.network(x)
        output = offset_normalized
        
        if mask is not None:
            output = output * mask
        
        return output

class TargetNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, y):
        self.mean = y.mean(axis=0)
        self.std = y.std(axis=0)
        self.std[self.std == 0] = 1.0
    
    def transform(self, y):
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer has not been fitted yet")
        return (y - self.mean) / self.std
    
    def inverse_transform(self, y):
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer has not been fitted yet")
        return y * self.std + self.mean
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)