# models/snn_layers.py
"""
Custom Spiking Neural Network layers and neurons
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
import numpy as np
from spikingjelly.activation_based import neuron, functional, layer, surrogate

class SpikingConv2d(nn.Module):
    """Spiking Convolutional layer with LIF neurons"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = False,
                 neuron_type: str = 'lif',
                 tau: float = 2.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 surrogate_function: Optional[Callable] = None,
                 detach_reset: bool = True):
        
        super().__init__()
        
        # Convolutional layer
        self.conv = layer.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        
        # Batch normalization
        self.bn = layer.BatchNorm2d(out_channels)
        
        # Spiking neuron
        if surrogate_function is None:
            surrogate_function = surrogate.ATan()
            
        if neuron_type == 'lif':
            self.neuron = neuron.LIFNode(
                tau=tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                surrogate_function=surrogate_function,
                detach_reset=detach_reset
            )
        elif neuron_type == 'plif':
            self.neuron = neuron.ParametricLIFNode(
                init_tau=tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                surrogate_function=surrogate_function,
                detach_reset=detach_reset
            )
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [T, B, C, H, W]
        x = self.conv(x)
        x = self.bn(x)
        x = self.neuron(x)
        return x

class SpikingLinear(nn.Module):
    """Spiking Linear layer with LIF neurons"""
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 neuron_type: str = 'lif',
                 tau: float = 2.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 surrogate_function: Optional[Callable] = None,
                 detach_reset: bool = True):
        
        super().__init__()
        
        self.fc = layer.Linear(in_features, out_features, bias=bias)
        
        if surrogate_function is None:
            surrogate_function = surrogate.ATan()
            
        if neuron_type == 'lif':
            self.neuron = neuron.LIFNode(
                tau=tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                surrogate_function=surrogate_function,
                detach_reset=detach_reset
            )
        else:
            self.neuron = neuron.ParametricLIFNode(
                init_tau=tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                surrogate_function=surrogate_function,
                detach_reset=detach_reset
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.neuron(x)
        return x

class AdaptiveSpikingPooling(nn.Module):
    """Adaptive pooling for spike trains"""
    
    def __init__(self, output_size: Tuple[int, int], mode: str = 'mean'):
        super().__init__()
        self.output_size = output_size
        self.mode = mode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        
        if self.mode == 'mean':
            # Average pooling over time then spatial
            x_time_pooled = x.mean(dim=0)  # [B, C, H, W]
            x_pooled = F.adaptive_avg_pool2d(x_time_pooled, self.output_size)
            # Repeat for all timesteps
            x_pooled = x_pooled.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        elif self.mode == 'max':
            # Max pooling
            x_reshaped = x.view(T * B, C, H, W)
            x_pooled = F.adaptive_max_pool2d(x_reshaped, self.output_size)
            x_pooled = x_pooled.view(T, B, C, *self.output_size)
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")
            
        return x_pooled

class ResidualSpikingBlock(nn.Module):
    """Residual block for SNNs with skip connections"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 neuron_type: str = 'lif'):
        super().__init__()
        
        self.conv1 = SpikingConv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, neuron_type=neuron_type
        )
        
        self.conv2 = layer.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = layer.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels, out_channels, kernel_size=1, 
                           stride=stride, bias=False),
                layer.BatchNorm2d(out_channels)
            )
        
        # Final neuron
        self.neuron = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.neuron(out)
        
        return out

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for spike trains"""
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [T, B, D]
        T, B, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(T, B, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 1, 3, 0, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(T, B, D)
        x = self.proj(x)
        
        return x