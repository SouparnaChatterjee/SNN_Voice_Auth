# models/speaker_head.py
"""
Speaker Verification head with metric learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class SpeakerVerificationHead(nn.Module):
    """Speaker embedding head with angular margin"""
    
    def __init__(self,
                 input_dim: int = 64,
                 embedding_dim: int = 128,
                 num_speakers: Optional[int] = None,
                 margin: float = 0.2,
                 scale: float = 30.0):
        
        super().__init__()
        
        # Embedding layers
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # For classification during training
        self.num_speakers = num_speakers
        if num_speakers is not None:
            self.classifier = ArcMarginProduct(
                embedding_dim, num_speakers, 
                margin=margin, scale=scale
            )
        
        self.margin = margin
        self.scale = scale
        
    def forward(self, 
                features: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Feature vectors from backbone [B, D]
            labels: Speaker labels for training [B]
        """
        # Generate embeddings
        embeddings = self.embedding_net(features)
        
        # L2 normalize
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        outputs = {
            'embeddings': embeddings_norm,
            'embeddings_raw': embeddings
        }
        
        # Classification (training only)
        if self.num_speakers is not None and labels is not None:
            logits = self.classifier(embeddings_norm, labels)
            outputs['logits'] = logits
            outputs['probabilities'] = F.softmax(logits, dim=-1)
        
        return outputs
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between embeddings"""
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        return F.cosine_similarity(emb1_norm, emb2_norm, dim=1)

class ArcMarginProduct(nn.Module):
    """ArcFace loss for angular margin"""
    
    def __init__(self, in_features: int, out_features: int, 
                 margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute cos(margin) and sin(margin) as tensors
        self.cos_m = torch.tensor(torch.cos(torch.tensor(margin)))
        self.sin_m = torch.tensor(torch.sin(torch.tensor(margin)))
        
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize weight and input
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        normalized_inputs = F.normalize(inputs, p=2, dim=1)
        
        # Compute cos(theta)
        cos_theta = F.linear(normalized_inputs, normalized_weight)
        cos_theta = cos_theta.clamp(-1, 1)
        
        # Compute sin(theta)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        
        # Compute cos(theta + m) using angle addition formula
        # cos(theta + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
        cos_theta_m = cos_theta * self.cos_m.to(cos_theta.device) - sin_theta * self.sin_m.to(sin_theta.device)
        
        # Create one-hot encoding
        one_hot = F.one_hot(labels, num_classes=self.out_features).float()
        
        # Select the appropriate cos values
        output = cos_theta * (1.0 - one_hot) + cos_theta_m * one_hot
        
        return output * self.scale