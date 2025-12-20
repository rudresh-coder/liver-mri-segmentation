import torch

EPS = 1e-7


def dice_coefficient(pred, target):
    """Binary Dice coefficient"""
    pred = pred.float()
    target = target.float()

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    dice = (2.0 * intersection) / union
    return dice.item()


def compute_confusion_matrix(pred, target):
    """Compute TP, FP, FN, TN for efficiency"""
    pred = pred.float()
    target = target.float()

    tp = torch.sum(pred * target)
    fp = torch.sum(pred * (1 - target))
    fn = torch.sum((1 - pred) * target)
    tn = torch.sum((1 - pred) * (1 - target))

    return tp, fp, fn, tn


def iou_score(pred, target):
    pred = pred.float()
    target = target.float()

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection

    iou = (intersection + EPS) / (union + EPS)
    return iou.item()


def precision_score(pred, target):
    tp, fp, fn, tn = compute_confusion_matrix(pred, target)
    precision = (tp + EPS) / (tp + fp + EPS)
    return precision.item()


def recall_score(pred, target):
    tp, fp, fn, tn = compute_confusion_matrix(pred, target)
    recall = (tp + EPS) / (tp + fn + EPS)
    return recall.item()


def f1_score(pred, target):
    tp, fp, fn, tn = compute_confusion_matrix(pred, target)
    precision = (tp + EPS) / (tp + fp + EPS)
    recall = (tp + EPS) / (tp + fn + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)
    return f1.item()