import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def batch_metrics(logits, targets, threshold=0.5):
    """Metrics pooled over all pixels of a batch. logits: (B, 2, H, W), targets: (B, H, W) in {0, 1}."""
    if isinstance(logits, list):
        logits = logits[0]
    pred = (torch.softmax(logits, dim=1)[:, 1] > threshold).float().reshape(-1)
    target = targets.reshape(-1)
    tp = torch.sum((pred == 1) & (target == 1)).float()
    fp = torch.sum((pred == 1) & (target == 0)).float()
    tn = torch.sum((pred == 0) & (target == 0)).float()
    fn = torch.sum((pred == 0) & (target == 1)).float()
    eps = 1e-8
    return {
        'dice': (2 * tp / (2 * tp + fp + fn + eps)).item(),
        'iou': (tp / (tp + fp + fn + eps)).item(),
        'precision': (tp / (tp + fp + eps)).item(),
        'recall': (tp / (tp + fn + eps)).item(),
        'accuracy': ((tp + tn) / (tp + fp + tn + fn + eps)).item(),
    }


def case_metrics(pred_mask, gt_mask):
    """Dice / IoU / Precision / Recall / ASD (pixels) of one binary prediction against its ground truth.
    Following the evaluation protocol of the paper, ASD is reported as 0 when either mask is empty."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    eps = 1e-7
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    if pred_mask.any() and gt_mask.any():
        surface_pred = pred_mask ^ binary_erosion(pred_mask)
        surface_gt = gt_mask ^ binary_erosion(gt_mask)
        dt_gt = distance_transform_edt(~surface_gt)
        dt_pred = distance_transform_edt(~surface_pred)
        asd = (dt_gt[surface_pred].mean() + dt_pred[surface_gt].mean()) / 2.0
    else:
        asd = 0.0
    return {
        'dice': 2 * tp / (2 * tp + fp + fn + eps),
        'iou': tp / (tp + fp + fn + eps),
        'precision': tp / (tp + fp + eps),
        'recall': tp / (tp + fn + eps),
        'asd': asd,
    }
