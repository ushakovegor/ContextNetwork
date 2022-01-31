import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianFocalLoss(nn.Module):
    """
    Focal loss for heatmaps.

    Parameters
    ----------
    alpha: float
        A balanced form for Focal loss.
    gamma: float
        The gamma for calculating the modulating factor.
    loss_weight:
        The weight for the Focal loss.
    """

    def __init__(self, alpha=2.0, gamma=4.0, loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred, gt):
        eps = 1e-12
        pos_weights = gt.eq(1)
        neg_weights = (1 - gt).pow(self.gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(self.alpha) * neg_weights
        loss = (pos_loss + neg_loss).sum()
        avg_factor = max(1, gt.eq(1).sum())
        return loss / avg_factor




def mse_loss(pred, target):
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')

class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss