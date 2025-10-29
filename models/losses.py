"""
Custom loss functions for dual-task SNN training
Includes FocalLoss, ArcFaceLoss, TripletLoss, CenterLoss, and CombinedLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ---------------------------------------------------------------------
# 1️⃣  Focal Loss (for Keyword Spotting)
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C]
            targets: [B] or [B, 1]
        """
        # ✅ Fix: squeeze if target has extra dimension
        if targets.ndim > 1:
            targets = targets.squeeze(-1)

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            # Move alpha to same device as targets
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            focal_loss = self.alpha[targets] * focal_loss

        return focal_loss.mean()


# ---------------------------------------------------------------------
# 2️⃣  ArcFace Loss (for Speaker Verification)
# ---------------------------------------------------------------------
class ArcFaceLoss(nn.Module):
    """Simplified ArcFace margin-based classification loss"""

    def __init__(self, scale: float = 30.0):
        super().__init__()
        self.scale = scale

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] - already scaled from ArcMarginProduct
            labels: [B] or [B, 1]
        """
        if labels.ndim > 1:
            labels = labels.squeeze(-1)

        # Logits from ArcMarginProduct are already scaled, use directly
        ce_loss = F.cross_entropy(logits, labels)
        return ce_loss


# ---------------------------------------------------------------------
# 3️⃣  Triplet Loss (for embedding separation)
# ---------------------------------------------------------------------
class TripletLoss(nn.Module):
    """Triplet loss for metric learning"""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: tensors of shape [N, D]
            where N is the number of triplets and D is embedding dimension
        """
        return self.loss_fn(anchor, positive, negative)


# ---------------------------------------------------------------------
# 4️⃣  Center Loss (for compact embeddings)
# ---------------------------------------------------------------------
class CenterLoss(nn.Module):
    """Center loss for intra-class compactness"""

    def __init__(self, num_classes: int, feat_dim: int, lambda_c: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] embeddings
            labels: [B] or [B, 1] class labels
        """
        if labels.ndim > 1:
            labels = labels.squeeze(-1)

        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        loss = ((features - centers_batch) ** 2).sum() / (2.0 * batch_size)
        return self.lambda_c * loss


# ---------------------------------------------------------------------
# 5️⃣  Combined Multi-Task Loss
# ---------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """Combines all task-specific losses"""

    def __init__(self,
                 alpha_kws: float = 1.0,
                 alpha_speaker: float = 1.0,
                 alpha_triplet: float = 0.5,
                 alpha_center: float = 0.5):
        super().__init__()

        self.kws_loss = FocalLoss()
        self.speaker_loss = ArcFaceLoss()
        self.triplet_loss = TripletLoss()
        self.center_loss = None  # Optional, can be set later

        self.alpha_kws = alpha_kws
        self.alpha_speaker = alpha_speaker
        self.alpha_triplet = alpha_triplet
        self.alpha_center = alpha_center

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dictionary containing task outputs
            targets: Dictionary containing target labels
        
        Returns:
            Dictionary of computed losses
        """
        total_loss = 0.0
        loss_dict = {}

        # ✅ Keyword Spotting loss
        if 'kws' in outputs and 'kws_labels' in targets:
            kws_loss = self.kws_loss(outputs['kws']['logits'], targets['kws_labels'])
            loss_dict['kws_loss'] = kws_loss
            total_loss += self.alpha_kws * kws_loss

        # ✅ Speaker classification loss
        if 'speaker' in outputs and 'speaker_labels' in targets:
            speaker_logits = outputs['speaker'].get('logits')
            if speaker_logits is not None:
                speaker_loss = self.speaker_loss(speaker_logits, targets['speaker_labels'])
                loss_dict['speaker_loss'] = speaker_loss
                total_loss += self.alpha_speaker * speaker_loss

        # ✅ Triplet loss (if embeddings are available)
        if 'speaker' in outputs and 'embeddings' in outputs['speaker']:
            emb = outputs['speaker']['embeddings']
            batch_size = emb.size(0)
            
            # ✅ FIX: Ensure we have valid, equal-sized triplets
            # We need at least 3 samples for a single triplet
            if batch_size >= 3:
                # Calculate how many complete triplets we can form
                num_triplets = batch_size // 3
                
                if num_triplets > 0:
                    # Slice embeddings into equal-sized anchor, positive, negative
                    # Each will have shape [num_triplets, embedding_dim]
                    anchor = emb[0:num_triplets]
                    positive = emb[num_triplets:2*num_triplets]
                    negative = emb[2*num_triplets:3*num_triplets]
                    
                    triplet_loss = self.triplet_loss(anchor, positive, negative)
                    loss_dict['triplet_loss'] = triplet_loss
                    total_loss += self.alpha_triplet * triplet_loss

        # ✅ Optional center loss
        if self.center_loss is not None and 'speaker_labels' in targets:
            if 'speaker' in outputs and 'embeddings' in outputs['speaker']:
                center_loss_val = self.center_loss(
                    outputs['speaker']['embeddings'],
                    targets['speaker_labels']
                )
                loss_dict['center_loss'] = center_loss_val
                total_loss += self.alpha_center * center_loss_val

        loss_dict['total_loss'] = total_loss
        return loss_dict