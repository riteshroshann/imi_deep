import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation block to calibrate feature importance."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S)
        global_pool = x.mean(dim=2)  # (B, C)
        attn = F.relu(self.fc1(global_pool))
        attn = torch.sigmoid(self.fc2(attn))  # (B, C)
        return x * attn.unsqueeze(2)

class SpatialTemporalAttention(nn.Module):
    """
    Improved Spatio-Temporal Attention (HybridSTA-V2) mechanism designed specifically 
    for the (N, 17, 16) CFRP tabular representation.
    
    Upgrades include:
    - Channel Attention (Squeeze-and-Excitation) prior to spatial mixing
    - Residual connections around the feature blocks
    - Increased capacity (d_model=128)
    """
    def __init__(
        self,
        n_sensors: int = 17,
        signal_length: int = 16,
        d_model: int = 128,  # Increased capacity
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 5,
        task: str = "rul",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.task = task
        self.signal_length = signal_length
        self.d_model = d_model
        
        # 1. Feature projection (1x1 Conv)
        self.feature_proj = nn.Sequential(
            nn.Conv1d(n_sensors, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Channel Attention (Feature calibration)
        self.channel_attn = ChannelAttention(d_model)
        
        # 3. Positional Encoding
        self.pos_emb = nn.Parameter(torch.randn(1, signal_length, d_model))
        
        # 4. Spatial Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 5. Global aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 6. Output Head
        output_dim = n_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        # B, 17, 16
        
        # 1. Project features
        feat_emb = self.feature_proj(x)  # (B, d_model, 16)
        
        # 2. Channel Attention
        feat_emb = self.channel_attn(feat_emb)  # (B, d_model, 16)
        
        # 3. Reshape for transformer: (B, 16, d_model)
        feat_emb = feat_emb.transpose(1, 2)
        
        # 4. Add spatial embedding
        transformer_input = feat_emb + self.pos_emb
        
        # 5. Spatial Attention & Residual Connection
        enc_out = self.spatial_encoder(transformer_input)  # (B, 16, d_model)
        enc_out = enc_out + transformer_input  # Residual skip
        
        # 6. Aggregate
        enc_out_t = enc_out.transpose(1, 2)
        global_repr = self.global_pool(enc_out_t).squeeze(-1)  # (B, d_model)
        
        # 7. Prediction
        out = self.head(global_repr)
        
        if self.task == "classification":
            return out
        else:
            return out.squeeze(-1)
