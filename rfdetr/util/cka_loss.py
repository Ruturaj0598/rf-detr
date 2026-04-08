"""
Multi-Layer CKA (Centered Kernel Alignment) Loss for RF-DETR
Implements hierarchical feature consistency regularization across multiple layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import numpy as np


# ============================================================
# AUGMENTATION UTILITIES
# ============================================================

class EnhancedAugmentation:
    """
    Enhanced augmentation strategies for CKA contrastive learning
    Implements diverse transformations for small object detection
    """
    
    @staticmethod
    def random_crop_with_scaling(image: torch.Tensor, scale_range=(0.8, 1.2)):
        """Random cropping with scaling for small object preservation"""
        B, C, H, W = image.shape
        scale = torch.empty(1).uniform_(*scale_range).item()
        new_h = max(1, int(H * scale))
        new_w = max(1, int(W * scale))
        
        # Resize
        resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Center crop or pad back to original size
        if new_h >= H and new_w >= W:
            top = (new_h - H) // 2
            left = (new_w - W) // 2
            output = resized[..., top:top+H, left:left+W]
        else:
            pad_h = max(0, H - new_h)
            pad_w = max(0, W - new_w)
            pad = (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2)
            output = F.pad(resized, pad, mode='reflect')
            output = output[..., :H, :W]
        
        return output.clamp(0.0, 1.0)
    
    @staticmethod
    def color_jitter(image: torch.Tensor, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """Apply color jittering"""
        # Brightness
        if torch.rand(1).item() < 0.5:
            factor = 1.0 + torch.empty(1).uniform_(-brightness, brightness).item()
            image = image * factor
        
        # Contrast
        if torch.rand(1).item() < 0.5:
            mean = image.mean(dim=[2, 3], keepdim=True)
            factor = 1.0 + torch.empty(1).uniform_(-contrast, contrast).item()
            image = (image - mean) * factor + mean
        
        return image.clamp(0.0, 1.0)
    
    @staticmethod
    def gaussian_blur(image: torch.Tensor, kernel_size=5, sigma_range=(0.1, 2.0)):
        """Apply Gaussian blur"""
        if torch.rand(1).item() < 0.5:
            sigma = torch.empty(1).uniform_(*sigma_range).item()
            # Simple blur implementation (can be replaced with torchvision.transforms.GaussianBlur)
            return F.avg_pool2d(image, kernel_size, stride=1, padding=kernel_size//2)
        return image
    
    @staticmethod
    def horizontal_flip(image: torch.Tensor, p=0.5):
        """Random horizontal flip"""
        if torch.rand(1).item() < p:
            return torch.flip(image, dims=[3])
        return image
    
    @staticmethod
    def create_augmented_view(image: torch.Tensor, augmentation_strength='moderate'):
        """
        Create augmented view with controllable strength
        
        Args:
            image: Input tensor [B, C, H, W]
            augmentation_strength: 'weak', 'moderate', 'strong'
        """
        view = image.clone()
        
        if augmentation_strength in ['moderate', 'strong']:
            view = EnhancedAugmentation.random_crop_with_scaling(view, scale_range=(0.85, 1.15))
        
        if augmentation_strength in ['moderate', 'strong']:
            view = EnhancedAugmentation.color_jitter(view, brightness=0.3, contrast=0.3)
        
        if augmentation_strength == 'strong':
            view = EnhancedAugmentation.gaussian_blur(view, sigma_range=(0.1, 2.0))
        
        view = EnhancedAugmentation.horizontal_flip(view, p=0.5)
        
        return view


# ============================================================
# LINEAR CKA COMPUTATION
# ============================================================

def linear_cka_from_features(feat_a: torch.Tensor, feat_b: torch.Tensor, eps=1e-8):
    """
    Compute linear CKA similarity between two feature tensors
    
    Args:
        feat_a: Feature tensor [B, C, H, W] or [B, N, C]
        feat_b: Feature tensor [B, C, H, W] or [B, N, C]
        eps: Small constant for numerical stability
    
    Returns:
        CKA similarity value (higher = more similar)
    """
    # Handle different input shapes
    if len(feat_a.shape) == 4:  # [B, C, H, W]
        B, C, H, W = feat_a.shape
        n = H * W
        # Reshape to [B, n, C]
        A = feat_a.view(B, C, n).permute(0, 2, 1).contiguous()
        Bf = feat_b.view(B, C, n).permute(0, 2, 1).contiguous()
    else:  # [B, N, C]
        A = feat_a
        Bf = feat_b
    
    # Flatten batch dimension for global CKA
    A = A.reshape(-1, A.size(-1))  # [B*n, C]
    Bf = Bf.reshape(-1, Bf.size(-1))  # [B*n, C]
    
    # Center the features
    A = A - A.mean(dim=0, keepdim=True)
    Bf = Bf - Bf.mean(dim=0, keepdim=True)
    
    # Compute Gram matrices
    K_A = torch.mm(A, A.t())  # [B*n, B*n]
    K_B = torch.mm(Bf, Bf.t())  # [B*n, B*n]
    
    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_ab = torch.sum(K_A * K_B)
    hsic_aa = torch.sum(K_A * K_A)
    hsic_bb = torch.sum(K_B * K_B)
    
    # Compute CKA
    cka = hsic_ab / (torch.sqrt(hsic_aa * hsic_bb) + eps)
    
    return torch.clamp(cka, 0.0, 1.0)


# ============================================================
# MULTI-LAYER CKA LOSS
# ============================================================

class MultiLayerCKALoss(nn.Module):
    """
    Multi-Layer CKA Loss for hierarchical feature consistency
    
    Aligns representations across multiple layers and augmented views
    """
    
    def __init__(
        self,
        layer_weights: Optional[Dict[int, float]] = None,
        progressive_schedule: bool = False,
        focus_on_small_objects: bool = True,
        temperature: float = 1.0
    ):
        """
        Args:
            layer_weights: Dict mapping layer indices to weights
                          e.g., {0: 2.0, 1: 1.5, 2: 1.0} emphasizes earlier layers
            progressive_schedule: If True, gradually increase CKA weight during training
            focus_on_small_objects: If True, emphasize higher-resolution features
            temperature: Temperature for scaling CKA similarities
        """
        super(MultiLayerCKALoss, self).__init__()
        
        self.layer_weights = layer_weights
        self.progressive_schedule = progressive_schedule
        self.focus_on_small_objects = focus_on_small_objects
        self.temperature = temperature
        
        # Progressive schedule parameters
        self.current_epoch = 0
        self.warmup_epochs = 5
        
    def set_epoch(self, epoch: int):
        """Update current epoch for progressive weighting"""
        self.current_epoch = epoch
    
    def get_progressive_weight(self) -> float:
        """Compute progressive weight based on current epoch"""
        if not self.progressive_schedule:
            return 1.0
        
        if self.current_epoch < self.warmup_epochs:
            return self.current_epoch / self.warmup_epochs
        else:
            return 1.0
    
    def compute_layer_weight(self, layer_idx: int, total_layers: int) -> float:
        """
        Compute weight for specific layer
        
        For small object detection, emphasize earlier (higher-resolution) layers
        """
        if self.layer_weights is not None and layer_idx in self.layer_weights:
            return self.layer_weights[layer_idx]
        
        if self.focus_on_small_objects:
            # Higher weight for earlier layers (higher resolution)
            # e.g., [2.0, 1.5, 1.0] for 3 layers
            return 2.0 - (layer_idx / max(total_layers - 1, 1))
        
        return 1.0
    
    def forward(
        self,
        features_view_a: Dict[int, torch.Tensor],
        features_view_b: Dict[int, torch.Tensor]
    ) -> tuple:
        """
        Compute multi-layer CKA loss
        
        Args:
            features_view_a: Dict of feature maps from view A
                            {layer_idx: feature_tensor [B, C, H, W]}
            features_view_b: Dict of feature maps from view B
        
        Returns:
            total_loss: Weighted sum of CKA losses across layers
            layer_losses: Dict of individual layer CKA losses (for logging)
        """
        layer_losses = {}
        total_loss = 0.0
        
        # Get layer indices (should be same for both views)
        layer_indices = sorted(features_view_a.keys())
        total_layers = len(layer_indices)
        
        # Compute CKA for each layer
        for layer_idx in layer_indices:
            feat_a = features_view_a[layer_idx]
            feat_b = features_view_b[layer_idx]
            
            # Ensure features have same spatial dimensions
            if feat_a.shape[-2:] != feat_b.shape[-2:]:
                feat_b = F.interpolate(
                    feat_b,
                    size=feat_a.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Compute CKA similarity
            cka_sim = linear_cka_from_features(feat_a, feat_b)
            
            # CKA loss = 1 - similarity (minimize dissimilarity)
            cka_loss = (1.0 - cka_sim) / self.temperature
            
            # Apply layer-specific weight
            layer_weight = self.compute_layer_weight(layer_idx, total_layers)
            weighted_loss = layer_weight * cka_loss
            
            # Accumulate
            total_loss += weighted_loss
            layer_losses[f'cka_layer_{layer_idx}'] = cka_loss.item()
            layer_losses[f'cka_layer_{layer_idx}_weighted'] = weighted_loss.item()
        
        # Apply progressive weighting
        progressive_weight = self.get_progressive_weight()
        total_loss = total_loss * progressive_weight
        
        # Average over number of layers
        total_loss = total_loss / total_layers
        
        # Add summary statistics
        layer_losses['cka_total'] = total_loss.item()
        layer_losses['progressive_weight'] = progressive_weight
        
        return total_loss, layer_losses


# ============================================================
# PAIRWISE MULTI-LAYER CKA
# ============================================================

class PairwiseMultiLayerCKA(nn.Module):
    """
    Computes pairwise CKA similarities across different layers
    Useful for understanding hierarchical relationships
    """
    
    def __init__(self, layer_pairs: Optional[List[tuple]] = None):
        """
        Args:
            layer_pairs: List of (layer_i, layer_j) tuples to compute CKA between
                        If None, computes all pairwise combinations
        """
        super(PairwiseMultiLayerCKA, self).__init__()
        self.layer_pairs = layer_pairs
    
    def forward(
        self,
        features_view_a: Dict[int, torch.Tensor],
        features_view_b: Dict[int, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute pairwise CKA across layers
        
        Returns:
            Dict of pairwise CKA values
        """
        layer_indices = sorted(features_view_a.keys())
        pairwise_cka = {}
        
        # Determine layer pairs
        if self.layer_pairs is None:
            # All pairwise combinations
            from itertools import combinations
            pairs = list(combinations(layer_indices, 2))
        else:
            pairs = self.layer_pairs
        
        for i, j in pairs:
            feat_a_i = features_view_a[i]
            feat_b_j = features_view_b[j]
            
            # Resize if needed
            if feat_a_i.shape[-2:] != feat_b_j.shape[-2:]:
                feat_b_j = F.interpolate(
                    feat_b_j,
                    size=feat_a_i.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            cka_sim = linear_cka_from_features(feat_a_i, feat_b_j)
            pairwise_cka[f'cka_layer_{i}_to_{j}'] = cka_sim.item()
        
        return pairwise_cka


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_multilayer_features(
    model,
    input_tensor: torch.Tensor,
    layer_names: Optional[List[str]] = None
) -> Dict[int, torch.Tensor]:
    """
    Extract features from multiple layers of the model
    
    This is a helper function that can be customized based on your model structure
    """
    # This function should be implemented based on your specific model
    # For RF-DETR, you already expose backbone_features in lwdetr.py
    pass


def log_cka_metrics(layer_losses: Dict[str, float], logger, step: int):
    """
    Log CKA metrics to tensorboard or other logging systems
    """
    for key, value in layer_losses.items():
        logger.add_scalar(f'CKA/{key}', value, step)
