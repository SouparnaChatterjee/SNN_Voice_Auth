# utils/metrics.py
"""
Evaluation metrics for both tasks
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Tuple, List

class AccuracyMetric:
    """Accuracy metric for keyword spotting"""
    
    def compute(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute accuracy"""
        return accuracy_score(labels, predictions)
    
    def compute_per_class(self, predictions: np.ndarray, 
                         labels: np.ndarray, 
                         class_names: List[str]) -> dict:
        """Compute per-class accuracy"""
        cm = confusion_matrix(labels, predictions)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        return {name: acc for name, acc in zip(class_names, per_class_acc)}

class EERMetric:
    """Equal Error Rate metric for speaker verification"""
    
    def compute(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute EER from embeddings"""
        # Generate positive and negative pairs
        pos_scores, neg_scores = self._generate_scores(embeddings, labels)
        
        # Compute EER
        eer = self._compute_eer(pos_scores, neg_scores)
        return eer
    
    def _generate_scores(self, embeddings: np.ndarray, 
                        labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate similarity scores for positive and negative pairs"""
        pos_scores = []
        neg_scores = []
        
        n_samples = len(embeddings)
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j])
                sim /= (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                
                if labels[i] == labels[j]:
                    pos_scores.append(sim)
                else:
                    neg_scores.append(sim)
        
        return np.array(pos_scores), np.array(neg_scores)
    
    def _compute_eer(self, pos_scores: np.ndarray, 
                    neg_scores: np.ndarray) -> float:
        """Compute Equal Error Rate"""
        # Create labels (1 for positive, 0 for negative)
        y_true = np.concatenate([np.ones_like(pos_scores), 
                                np.zeros_like(neg_scores)])
        y_scores = np.concatenate([pos_scores, neg_scores])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Find EER
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer