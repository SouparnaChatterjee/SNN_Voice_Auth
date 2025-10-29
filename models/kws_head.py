# models/kws_head.py
"""
Keyword Spotting classification head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class KeywordSpottingHead(nn.Module):
    """Classification head for keyword detection"""
    
    def __init__(self,
                 input_dim: int = 64,
                 num_keywords: int = 6,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.2):
        
        super().__init__()
        
        if hidden_dim is None:
            # Direct classification
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_keywords)
            )
        else:
            # With hidden layer
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_keywords)
            )
        
        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Feature vectors from backbone [B, D]
        Returns:
            Dictionary with logits and probabilities
        """
        logits = self.classifier(features)
        
        # Temperature scaling for better calibration
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        return {
            'logits': logits,
            'scaled_logits': scaled_logits,
            'probabilities': probs,
            'predictions': torch.argmax(probs, dim=-1)
        }