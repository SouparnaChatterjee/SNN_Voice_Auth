# models/dual_task_snn.py
"""
Combined SNN model for keyword spotting and speaker verification
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from models.snn_backbone import LightweightSNNBackbone, SNNBackbone
from models.kws_head import KeywordSpottingHead
from models.speaker_head import SpeakerVerificationHead

class DualTaskSNN(nn.Module):
    """SNN for both keyword spotting and speaker verification"""
    
    def __init__(self,
                 input_dim: int = 13,
                 time_steps: int = 100,
                 num_keywords: int = 6,
                 num_speakers: Optional[int] = None,
                 backbone_type: str = 'lightweight',
                 shared_features: bool = True):
        
        super().__init__()
        
        self.shared_features = shared_features
        self.time_steps = time_steps
        
        # Backbone
        if backbone_type == 'lightweight':
            self.backbone = LightweightSNNBackbone(
                input_dim=input_dim,
                hidden_dim=128,
                output_dim=64,
                time_steps=time_steps
            )
            feature_dim = 64
        else:
            self.backbone = SNNBackbone(
                input_channels=1,
                base_channels=32,
                num_classes=128,
                time_steps=time_steps
            )
            feature_dim = 128
        
        # Task-specific heads
        if not shared_features:
            # Separate feature extractors
            self.kws_proj = nn.Linear(feature_dim, feature_dim)
            self.speaker_proj = nn.Linear(feature_dim, feature_dim)
        
        self.kws_head = KeywordSpottingHead(
            input_dim=feature_dim,
            num_keywords=num_keywords
        )
        
        self.speaker_head = SpeakerVerificationHead(
            input_dim=feature_dim,
            embedding_dim=128,
            num_speakers=num_speakers
        )
    
    def forward(self,
                x: torch.Tensor,
                task: str = 'both',
                labels: Optional[torch.Tensor] = None) -> Dict[str, Dict]:
        """
        Args:
            x: Input spike trains [B, F, T]
            task: 'kws', 'speaker', or 'both'
            labels: Task-specific labels
        """
        # Extract features
        backbone_out = self.backbone(x)
        features = backbone_out['features']
        
        outputs = {'backbone': backbone_out}
        
        # Task-specific processing
        if task in ['kws', 'both']:
            kws_features = features
            if not self.shared_features:
                kws_features = self.kws_proj(features)
            
            kws_out = self.kws_head(kws_features)
            outputs['kws'] = kws_out
        
        if task in ['speaker', 'both']:
            speaker_features = features
            if not self.shared_features:
                speaker_features = self.speaker_proj(features)
            
            speaker_out = self.speaker_head(speaker_features, labels)
            outputs['speaker'] = speaker_out
        
        return outputs
    
    def extract_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings for both tasks"""
        backbone_out = self.backbone(x)
        features = backbone_out['features']
        
        # Get speaker embeddings
        speaker_features = features
        if not self.shared_features:
            speaker_features = self.speaker_proj(features)
        speaker_out = self.speaker_head(speaker_features)
        
        return features, speaker_out['embeddings']