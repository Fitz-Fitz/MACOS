import torch
import torch.nn as nn
import torch.nn.functional as F
from models.motion_unet import warp


def _one_hot(targets, num_classes):
    one_hot = torch.zeros(targets.shape[0], num_classes, *targets.shape[1:], device=targets.device)
    return one_hot.scatter_(1, targets.long().unsqueeze(1), 1)


class DiceLoss(nn.Module):
    """Soft Dice loss of the foreground class. logits: (N, C, H, W), targets: (N, H, W)."""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = _one_hot(targets, probs.shape[1])
        intersection = torch.sum(probs * one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(one_hot, dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice[:, 1].mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = _one_hot(targets, probs.shape[1])
        loss = 0
        for c in range(probs.shape[1]):
            p_t = probs[:, c] * one_hot[:, c] + (1 - probs[:, c]) * (1 - one_hot[:, c])
            alpha_t = self.alpha if c == 1 else (1 - self.alpha)
            loss = loss + alpha_t * (1 - p_t + self.smooth) ** self.gamma * (-torch.log(p_t + self.smooth))
        return loss.mean()


class SegmentationLoss(nn.Module):
    """CE + Dice (+ focal) for the key frame. Accepts logits (N, C, H, W) or, with deep supervision,
    a list [main, ds1, ds2, ...] weighted by ds_weights. Returns (total, ce, dice)."""

    def __init__(self, ce_weight=1.0, dice_weight=1.0, focal_weight=0.0, focal_alpha=0.25, focal_gamma=2.0,
                 ds_weights=(1.0, 0.5, 0.25, 0.125, 0.0625)):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ds_weights = ds_weights
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(focal_alpha, focal_gamma) if focal_weight > 0 else None

    def _single(self, logits, targets):
        ce = self.ce(logits, targets.long())
        dice = self.dice(logits, targets)
        total = self.ce_weight * ce + self.dice_weight * dice
        if self.focal is not None:
            total = total + self.focal_weight * self.focal(logits, targets)
        return total, ce, dice

    def forward(self, logits, targets):
        if not isinstance(logits, (list, tuple)):
            return self._single(logits, targets)
        total = ce = dice = 0
        for i, head in enumerate(logits):
            w = self.ds_weights[i] if i < len(self.ds_weights) else 0.1
            t, c, d = self._single(head, targets)
            total, ce, dice = total + w * t, ce + w * c, dice + w * d
        return total, ce, dice


def masked_ncc_loss(warped, fixed, mask=None):
    """1 - NCC between warped and fixed images, computed per sample over the pixels where mask > 0."""
    values = []
    for i in range(warped.shape[0]):
        if mask is not None:
            keep = mask[i] > 0
            if keep.sum() == 0:
                values.append(torch.tensor(0.0, device=warped.device))
                continue
            w, f = warped[i][:, keep].flatten(), fixed[i][:, keep].flatten()
        else:
            w, f = warped[i].flatten(), fixed[i].flatten()
        w, f = w - w.mean(), f - f.mean()
        values.append((w * f).sum() / (torch.sqrt((w ** 2).sum() * (f ** 2).sum()) + 1e-8))
    return 1.0 - torch.stack(values).mean()


def smoothness_loss(flow, mask=None):
    """Squared first-order finite differences of the displacement field, optionally restricted to mask > 0."""
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    if mask is None:
        return torch.mean(dx ** 2) + torch.mean(dy ** 2)
    mask_dy = (mask[:, 1:, :] * mask[:, :-1, :]).unsqueeze(1).expand_as(dy)
    mask_dx = (mask[:, :, 1:] * mask[:, :, :-1]).unsqueeze(1).expand_as(dx)
    dy_loss = torch.sum((dy * mask_dy) ** 2) / (torch.sum(mask_dy) + 1e-8)
    dx_loss = torch.sum((dx * mask_dx) ** 2) / (torch.sum(mask_dx) + 1e-8)
    return dx_loss + dy_loss


class MotionLoss(nn.Module):
    """Global Motion Loss: photometric (NCC) consistency between the warped previous frame and the current
    frame on predicted background, plus smoothness of the displacement field.
    Returns (total, similarity, smoothness)."""

    def __init__(self, similarity_weight=1.0, smoothness_weight=0.05, vessel_threshold=0.5, min_mask_ratio=0.0):
        super().__init__()
        self.similarity_weight = similarity_weight
        self.smoothness_weight = smoothness_weight
        self.vessel_threshold = vessel_threshold
        self.min_mask_ratio = min_mask_ratio

    def background_mask(self, logits):
        """logits: (B, T, C, H, W) sequence predictions -> (B*(T-1), H, W) background mask of frames 1..T-1."""
        vessel_prob = torch.softmax(logits[:, 1:], dim=2)[:, :, 1]
        mask = (vessel_prob < self.vessel_threshold).float()
        if mask.mean() < self.min_mask_ratio:
            mask = (vessel_prob < min(self.vessel_threshold + 0.2, 0.9)).float()
        return mask.reshape(-1, *mask.shape[2:])

    def forward(self, warped, fixed, flow, logits=None):
        if logits is not None:
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            mask = self.background_mask(logits)
            assert mask.shape[0] == warped.shape[0], f'{mask.shape[0]} masks for {warped.shape[0]} frame pairs'
            similarity = masked_ncc_loss(warped, fixed, mask)
            smoothness = smoothness_loss(flow, mask) + smoothness_loss(flow, 1 - mask)
        else:
            similarity = masked_ncc_loss(warped, fixed)
            smoothness = smoothness_loss(flow)
        return self.similarity_weight * similarity + self.smoothness_weight * smoothness, similarity, smoothness


class PhysicsLoss(nn.Module):
    """Physics-Informed Vessel Loss (monotonic vessel growth during contrast wash-in).

    For every transition t -> t+1 with t >= 1 (the transition out of the pre-injection frame 0 is skipped)
    the vessel map P_t is warped onto frame t+1 with the estimated displacement field. The loss penalises
    vessel response that disappears after warping (leakage), rewards the overlap with P_{t+1} (recall) and,
    with a small weight, the change of total vessel mass. The vessel maps are detached, so the gradient
    reaches the displacement field only.
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, flow):
        """logits: (B, T, C, H, W) sequence predictions, flow: (B, T-1, 2, H, W)."""
        if flow is None or flow.shape[1] == 0:
            return torch.tensor(0.0, device=logits.device)
        assert flow.shape[-2:] == logits.shape[-2:], 'displacement field and predictions must have the same resolution'
        vessel = torch.softmax(logits, dim=2)[:, :, 1].detach()
        total, steps = 0.0, 0
        for t in range(1, logits.shape[1] - 1):
            prev, cur = vessel[:, t], vessel[:, t + 1]
            prev_sum = prev.sum()
            warped_prev = warp(prev.unsqueeze(1), flow[:, t]).squeeze(1)
            leakage = F.relu(warped_prev - cur).sum() / (prev_sum + 1e-6)
            recall = torch.min(warped_prev, cur).sum() / (prev_sum + 1e-6)
            mass = torch.abs(warped_prev.sum() - prev_sum) / (prev_sum + 1e-6)
            total = total + self.alpha * leakage + self.beta * (1.0 - recall) + self.gamma * mass
            steps += 1
        if steps == 0:
            return torch.tensor(0.0, device=logits.device)
        return total / steps
