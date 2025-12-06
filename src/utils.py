# src/utils.py
import os
import torch
import numpy as np

def dice_coef(pred, target, smooth=1e-6):
    """
    pred, target: tensors of shape (B, 1, H, W) with values in [0,1]
    """
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice.mean()

def dice_loss(pred, target, smooth=1e-6):
    return 1.0 - dice_coef(pred, target, smooth)

def iou_score(pred, target, smooth=1e-6):
    """
    Intersection over Union metric.
    """
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict()
    }, path)

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
