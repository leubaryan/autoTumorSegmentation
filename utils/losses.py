import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for medical image segmentation"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal loss
        pt = inputs * targets + (1 - inputs) * (1 - targets)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss function (Dice + BCE)"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        # Dice loss
        dice_loss = DiceLoss(smooth=self.smooth)(inputs, targets)
        
        # BCE loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return total_loss


class TverskyLoss(nn.Module):
    """Tversky Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        fps = (inputs * (1 - targets)).sum()
        fns = ((1 - inputs) * targets).sum()
        
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fps + self.beta * fns + self.smooth)
        
        return 1 - tversky


class HausdorffLoss(nn.Module):
    """Hausdorff Distance Loss (approximation)"""
    
    def __init__(self, alpha=2.0, beta=2.0):
        super(HausdorffLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Calculate distance maps
        pred_dist = self._distance_map(inputs)
        target_dist = self._distance_map(targets)
        
        # Calculate Hausdorff loss
        loss = torch.mean(torch.abs(pred_dist - target_dist))
        
        return loss
    
    def _distance_map(self, tensor):
        """Calculate distance map for binary tensor"""
        # This is a simplified version - in practice, you might want to use
        # scipy.ndimage.distance_transform_edt or similar
        return tensor


class BoundaryLoss(nn.Module):
    """Boundary Loss for better boundary preservation"""
    
    def __init__(self, weight=1.0):
        super(BoundaryLoss, self).__init__()
        self.weight = weight
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Calculate gradients
        grad_x_pred = torch.abs(inputs[:, :, :, 1:] - inputs[:, :, :, :-1])
        grad_y_pred = torch.abs(inputs[:, :, 1:, :] - inputs[:, :, :-1, :])
        
        grad_x_target = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])
        grad_y_target = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
        
        # Calculate boundary loss
        boundary_loss = torch.mean(torch.abs(grad_x_pred - grad_x_target)) + \
                       torch.mean(torch.abs(grad_y_pred - grad_y_target))
        
        return self.weight * boundary_loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss"""
    
    def __init__(self, pos_weight=2.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Calculate weighted BCE loss
        loss = -(self.pos_weight * targets * torch.log(inputs + 1e-8) + 
                 (1 - targets) * torch.log(1 - inputs + 1e-8))
        
        return loss.mean()


def get_loss_function(loss_name, **kwargs):
    """Factory function to get loss function by name"""
    loss_functions = {
        'dice': DiceLoss,
        'focal': FocalLoss,
        'combined': CombinedLoss,
        'tversky': TverskyLoss,
        'hausdorff': HausdorffLoss,
        'boundary': BoundaryLoss,
        'weighted_bce': WeightedBCELoss,
        'bce': nn.BCEWithLogitsLoss
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)


if __name__ == "__main__":
    # Test loss functions
    batch_size = 2
    height, width = 256, 256
    
    # Create dummy data
    inputs = torch.randn(batch_size, 1, height, width)
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Test different loss functions
    losses = {
        'Dice': DiceLoss(),
        'Focal': FocalLoss(),
        'Combined': CombinedLoss(),
        'Tversky': TverskyLoss(),
        'Boundary': BoundaryLoss(),
        'Weighted BCE': WeightedBCELoss()
    }
    
    print("Testing loss functions:")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(inputs, targets)
        print(f"{name} Loss: {loss_value.item():.4f}") 