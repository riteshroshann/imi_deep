import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTemporalAttention(nn.Module):
    """
    Novel Spatio-Temporal Attention (STA) mechanism designed specifically 
    for the (N, 17, 16) CFRP tabular representation where:
    - 17 = channels (frequency/time domain feature descriptors)
    - 16 = sequence (spatial sensor paths in the 4x4 array)
    
    This architecture explicitly separates feature-wise representation learning 
    (1x1 Convs) from cross-path spatial attention (Transformer Encoder),
    making it highly effective for multi-sensor damage localization.
    """
    def __init__(
        self,
        n_sensors: int = 17,
        signal_length: int = 16,
        d_model: int = 64,
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
        
        # 1. Feature projection: projects the 17 raw statistical features 
        # into a higher-dimensional embedding space per sensor path independently.
        # Input: (B, 17, 16) -> (B, d_model, 16)
        self.feature_proj = nn.Sequential(
            nn.Conv1d(n_sensors, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Positional Encoding: injects spatial structure (which path is which)
        self.pos_emb = nn.Parameter(torch.randn(1, signal_length, d_model))
        
        # 3. Spatial Attention: computes relationships between different paths
        # (e.g., path 1-7 vs path 4-16) to localize damage.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Global aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 5. Output Head
        output_dim = n_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        # x shape: (B, 17, 16)
        B, C, S = x.shape
        
        # 1. Project features
        feat_emb = self.feature_proj(x)  # (B, d_model, 16)
        
        # 2. Reshape for transformer: (B, S, d_model)
        feat_emb = feat_emb.transpose(1, 2)
        
        # 3. Add spatial embedding
        feat_emb = feat_emb + self.pos_emb
        
        # 4. Spatial Attention (cross-sensor interaction)
        enc_out = self.spatial_encoder(feat_emb)  # (B, S, d_model)
        
        # 5. Aggregate across all spatial paths
        # Transpose back to (B, d_model, S) for pooling
        enc_out_t = enc_out.transpose(1, 2)
        global_repr = self.global_pool(enc_out_t).squeeze(-1)  # (B, d_model)
        
        # 6. Prediction
        out = self.head(global_repr)
        
        if self.task == "classification":
            return out
        else:
            return out.squeeze(-1)
