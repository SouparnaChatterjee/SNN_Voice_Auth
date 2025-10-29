# models/snn_backbone.py
"""
Core SNN backbone for feature extraction
"""
import torch
import torch.nn as nn
from typing import Dict
from spikingjelly.activation_based import functional

from models.snn_layers import (
    SpikingConv2d, SpikingLinear, ResidualSpikingBlock,
    AdaptiveSpikingPooling, TemporalAttention
)


class SNNBackbone(nn.Module):
    """Spiking CNN backbone for audio feature extraction"""

    def __init__(self,
                 input_channels: int = 1,
                 base_channels: int = 32,
                 num_classes: int = 128,  # Feature dimension
                 time_steps: int = 100,
                 neuron_type: str = 'lif',
                 use_attention: bool = True):

        super().__init__()
        self.time_steps = time_steps
        self.use_attention = use_attention

        # Initial spike encoding layer (no neurons, just projection)
        self.input_proj = nn.Conv2d(
            input_channels, base_channels,
            kernel_size=5, stride=1, padding=2
        )

        # Spiking CNN layers
        self.layer1 = self._make_layer(
            base_channels, base_channels * 2,
            num_blocks=2, stride=2, neuron_type=neuron_type
        )

        self.layer2 = self._make_layer(
            base_channels * 2, base_channels * 4,
            num_blocks=2, stride=2, neuron_type=neuron_type
        )

        self.layer3 = self._make_layer(
            base_channels * 4, base_channels * 8,
            num_blocks=2, stride=2, neuron_type=neuron_type
        )

        # Global pooling
        self.global_pool = AdaptiveSpikingPooling((1, 1), mode='mean')

        # Feature projection
        self.feature_dim = base_channels * 8
        self.feature_proj = SpikingLinear(
            self.feature_dim, num_classes,
            neuron_type=neuron_type
        )

        # Optional temporal attention
        if use_attention:
            self.temporal_attn = TemporalAttention(num_classes)

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int, neuron_type: str) -> nn.Sequential:
        """Create a layer with multiple residual blocks"""
        layers = []

        # First block with stride
        layers.append(
            ResidualSpikingBlock(in_channels, out_channels,
                                 stride=stride, neuron_type=neuron_type)
        )

        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(
                ResidualSpikingBlock(out_channels, out_channels,
                                     stride=1, neuron_type=neuron_type)
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            x: Input spike trains [B, C, F, T] where F is features, T is time
        Returns:
            dict with 'features' and 'spikes'
        """
        B, C, F, T = x.shape

        # Reshape to [T, B, C, H, W] for SpikingJelly
        x = x.permute(3, 0, 1, 2).contiguous()  # [T, B, C, F]
        x = x.unsqueeze(-1)  # [T, B, C, F, 1]

        # Initial projection
        x_seq = []
        for t in range(T):
            x_t = self.input_proj(x[t])
            x_seq.append(x_t)
        x = torch.stack(x_seq, dim=0)  # [T, B, C, H, W]

        # Feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global pooling
        x = self.global_pool(x)  # [T, B, C, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [T, B, C]

        # Feature projection
        features = self.feature_proj(x)  # [T, B, num_classes]

        # Temporal attention
        if self.use_attention:
            features = features + self.temporal_attn(features)

        # Aggregate over time
        features_mean = features.mean(dim=0)  # [B, num_classes]

        # Reset neurons for next forward
        functional.reset_net(self)

        return {
            'features': features_mean,
            'temporal_features': features,
            'spike_rates': features.mean(dim=0)
        }


class LightweightSNNBackbone(nn.Module):
    """Lighter SNN backbone for faster CPU inference"""

    def __init__(self,
                 input_dim: int = 13,  # MFCC features
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 time_steps: int = 100,
                 num_layers: int = 3,
                 neuron_type: str = 'lif'):

        super().__init__()
        self.time_steps = time_steps

        # Build sequential layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(
                SpikingLinear(
                    current_dim, next_dim,
                    neuron_type=neuron_type,
                    tau=2.0 + i * 0.5  # Varying time constants
                )
            )
            current_dim = next_dim

            # Add dropout for regularization
            if i < num_layers - 1:
                layers.append(nn.Dropout(0.2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input spikes [B, F, T] where F is features, T is time
        """
        B, F, T = x.shape

        # Reshape for time-first processing
        x = x.permute(2, 0, 1)  # [T, B, F]

        # Process through layers
        features = self.layers(x)  # [T, B, output_dim]

        # Aggregate features
        features_mean = features.mean(dim=0)  # [B, output_dim]
        features_max = features.max(dim=0)[0]  # [B, output_dim]

        # Reset neurons
        functional.reset_net(self)

        return {
            'features': features_mean,
            'features_max': features_max,
            'temporal_features': features,
            'spike_rates': (features > 0).float().mean(dim=0)
        }
