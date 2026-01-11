import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

class OFCDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None
            
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class SpectralAttention(nn.Module):
    def __init__(self, channels):
        super(SpectralAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x):
        B, C = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).reshape(B, C)
        out = self.proj(out)
        return out


class SpectralBranch(nn.Module):
    def __init__(self, input_channels=95, hidden_dim=128):
        super(SpectralBranch, self).__init__()
        
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        ])
        
        self.attention = SpectralAttention(hidden_dim)
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads=4)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = x.squeeze(-1)
        x = self.attention(x)
        x = self.self_attn(x)
        
        return x


class ScalarBranch(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super(ScalarBranch, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class CrossModalFusion(nn.Module):
    def __init__(self, spectral_dim, scalar_dim, fusion_dim=256):
        super(CrossModalFusion, self).__init__()
        
        self.spectral_proj = nn.Linear(spectral_dim, fusion_dim)
        self.scalar_proj = nn.Linear(scalar_dim, fusion_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads=4, batch_first=True)
        
    def forward(self, spectral_feat, scalar_feat):
        spec_proj = self.spectral_proj(spectral_feat)
        scal_proj = self.scalar_proj(scalar_feat)
        
        concat = torch.cat([spec_proj, scal_proj], dim=1)
        fused = self.fusion(concat)
        
        query = fused.unsqueeze(1)
        key = value = fused.unsqueeze(1)
        
        attn_out, _ = self.cross_attn(query, key, value)
        attn_out = attn_out.squeeze(1)
        
        return fused + attn_out


class GainPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=95, 
                 num_spectral=95, num_scalar=4, num_cat=3, num_mask=95):
        super(GainPredictor, self).__init__()
        
        self.num_spectral = num_spectral
        self.num_scalar = num_scalar
        self.num_cat = num_cat
        self.num_mask = num_mask
        
        spectral_input_dim = num_spectral
        scalar_input_dim = num_scalar + num_cat + num_mask
        
        self.spectral_branch = SpectralBranch(spectral_input_dim, hidden_dim=128)
        self.scalar_branch = ScalarBranch(scalar_input_dim, hidden_dims=[256, 128])
        self.fusion = CrossModalFusion(128, 128, fusion_dim=256)
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        spectral_input = x[:, :self.num_spectral]
        scalar_input = x[:, self.num_spectral:]
        
        spectral_feat = self.spectral_branch(spectral_input)
        scalar_feat = self.scalar_branch(scalar_input)
        
        fused = self.fusion(spectral_feat, scalar_feat)
        
        output = self.decoder(fused)
        
        return output
