"""
Masked Marginal & Exclusion Loss Formulations for Partially Labeled Multi-Dataset Training.

Crucial for training on heterogeneous public datasets (AMOS, TotalSegmentator, CHAOS)
where specific structures are missing from annotations without falsely penalizing the model.
"""

from typing import List, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedDiceLoss(nn.Module):
    """
    Computes Soft Dice loss only over known/present classes for each volumetric batch item.
    Excludes unannotated structures to prevent false-negative gradient penalty.
    """
    def __init__(
        self,
        smooth: float = 1e-5,
        include_background: bool = False,
        squared_pred: bool = True
    ):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.squared_pred = squared_pred

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        present_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw unnormalized network predictions.
            targets: (B, 1, D, H, W) or (B, D, H, W) integer label maps [0..C-1].
            present_mask: (B, C) boolean tensor indicating which classes are annotated.
        """
        if targets.ndim == logits.ndim - 1:
            targets = targets.unsqueeze(1)
            
        b, c, *dims = logits.shape
        probs = F.softmax(logits, dim=1)
        
        # Convert integer targets to one-hot encoding: (B, C, D, H, W)
        target_one_hot = torch.zeros_like(probs)
        target_one_hot.scatter_(1, targets.long(), 1.0)
        
        # Determine active classes
        start_idx = 0 if self.include_background else 1
        
        # Flatten spatial dimensions: (B, C, N)
        probs_flat = probs[:, start_idx:].flatten(2)
        target_flat = target_one_hot[:, start_idx:].flatten(2)
        
        # Intersection & cardinality
        intersection = torch.sum(probs_flat * target_flat, dim=-1)
        if self.squared_pred:
            cardinality = torch.sum(probs_flat ** 2, dim=-1) + torch.sum(target_flat ** 2, dim=-1)
        else:
            cardinality = torch.sum(probs_flat, dim=-1) + torch.sum(target_flat, dim=-1)
            
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score  # (B, C_fg)
        
        if present_mask is not None:
            # present_mask is (B, C). Slice to match foreground
            active_mask = present_mask[:, start_idx:].float()
            # Mean over active classes only
            loss = torch.sum(dice_loss * active_mask) / (torch.sum(active_mask) + 1e-8)
        else:
            loss = torch.mean(dice_loss)
            
        return loss


class MaskedCrossEntropyLoss(nn.Module):
    """
    Computes Cross-Entropy loss marginalized over present classes.
    Re-normalizes logits across only known classes per volume so unannotated organs
    are not forced into background.
    """
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        present_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W)
            targets: (B, 1, D, H, W) or (B, D, H, W)
            present_mask: (B, C) binary/boolean tensor
        """
        if targets.ndim == logits.ndim:
            targets = targets.squeeze(1)
        targets = targets.long()
        
        if present_mask is None:
            return F.cross_entropy(logits, targets, weight=self.weight)
        
        batch_losses = []
        b, c, *dims = logits.shape
        
        for i in range(b):
            item_present = present_mask[i]  # (C,)
            active_indices = torch.where(item_present)[0]
            
            # Sub-select logits for known classes
            item_logits = logits[i:i+1, active_indices]  # (1, C_known, D, H, W)
            item_target = targets[i:i+1]                 # (1, D, H, W)
            
            # Remap target indices to the active subset
            remapped_target = torch.zeros_like(item_target)
            for new_idx, orig_idx in enumerate(active_indices):
                remapped_target[item_target == orig_idx] = new_idx
                
            item_loss = F.cross_entropy(item_logits, remapped_target)
            batch_losses.append(item_loss)
            
        return torch.stack(batch_losses).mean()


class HarmonizedCrossScanLoss(nn.Module):
    """
    Composite loss combining Masked Soft Dice and Masked Cross-Entropy.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background_dice: bool = False
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = MaskedDiceLoss(include_background=include_background_dice)
        self.ce_loss = MaskedCrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        present_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss_dice = self.dice_loss(logits, targets, present_mask=present_mask)
        loss_ce = self.ce_loss(logits, targets, present_mask=present_mask)
        return self.dice_weight * loss_dice + self.ce_weight * loss_ce

