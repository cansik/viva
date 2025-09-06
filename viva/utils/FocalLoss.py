import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss for multi-class classification.

        Parameters:
            alpha (float or tensor): Weighting factor for the loss (can be set per class as a tensor).
            gamma (float): Focusing parameter that reduces the loss for well-classified examples.
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Parameters:
            inputs: raw logits tensor with shape (batch_size, num_classes)
            targets: ground truth labels (long tensor) with shape (batch_size,)
        """
        # Compute the standard cross entropy loss (per sample, without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Convert cross-entropy loss into the probability of the true class:
        pt = torch.exp(-ce_loss)  # pt = exp(-loss) gives p_t for each sample

        # Compute the focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
