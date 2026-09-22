import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import Encoder, TemporalDecoder, FEATURES


def warp(src, flow, mode='bilinear'):
    """Bilinear spatial warping of src (N, C, H, W) with a displacement field flow (N, 2, H, W).
    flow[:, 0] is the displacement along H (y) and flow[:, 1] along W (x), in pixels."""
    H, W = flow.shape[2:]
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=flow.device, dtype=torch.float32),
                                    torch.arange(W, device=flow.device, dtype=torch.float32), indexing='ij')
    locs = torch.stack([grid_y, grid_x]).unsqueeze(0) + flow
    locs = torch.stack([2 * (locs[:, 1] / (W - 1) - 0.5), 2 * (locs[:, 0] / (H - 1) - 0.5)], dim=-1)
    return F.grid_sample(src, locs, mode=mode, align_corners=True)


class MotionAwareGRUCell(nn.Module):
    """MA-GRU cell: estimates the displacement between the previous and the current frame feature,
    warps the previous hidden state onto the current frame, then applies the GRU gates."""

    def __init__(self, channels, dropout=0.1):
        super().__init__()
        groups = max(1, min(8, channels // 4))

        def gate():
            return nn.Sequential(
                nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels, channels, 1, bias=True),
            )

        self.reset_gate_conv = gate()
        self.update_gate_conv = gate()
        self.candidate_conv = gate()
        self.residual_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.residual_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        self.motion_estimator = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GroupNorm(max(1, min(8, channels // 8)), channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // 2, 2, 3, padding=1),
        )
        self.dropout = nn.Dropout2d(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.update_gate_conv[-1].bias, 0.5)
        nn.init.constant_(self.reset_gate_conv[-1].bias, 0.0)
        nn.init.constant_(self.motion_estimator[-1].weight, 0)
        nn.init.constant_(self.motion_estimator[-1].bias, 0)

    def forward(self, x, h_prev, x_prev):
        displacement = self.motion_estimator(torch.cat([x, x_prev], dim=1))
        h_aligned = warp(h_prev, displacement)
        combined = torch.cat([x, h_aligned], dim=1)
        reset_gate = torch.sigmoid(self.reset_gate_conv(combined))
        update_gate = torch.sigmoid(self.update_gate_conv(combined))
        candidate = torch.tanh(self.candidate_conv(torch.cat([x, reset_gate * h_aligned], dim=1)))
        h_new = (1 - update_gate) * candidate + update_gate * h_aligned
        h_new = h_new + self.residual_proj(x) * self.residual_gate(x)
        if self.training:
            h_new = self.dropout(h_new)
        return h_new, displacement


class MotionAwareGRU(nn.Module):
    """Runs an MA-GRU cell over a feature sequence (B, T, C, H, W).
    Returns the hidden states (B, T, C, H, W) and the displacement fields (B, T-1, 2, H, W),
    where entry t is the motion from frame t to frame t+1."""

    def __init__(self, channels):
        super().__init__()
        self.gru_cells = MotionAwareGRUCell(channels)

    def forward(self, x):
        B, T, C, H, W = x.shape
        outputs, displacements = [], []
        h, _ = self.gru_cells(x[:, 0], torch.zeros_like(x[:, 0]), x[:, 0])
        outputs.append(h)
        for t in range(1, T):
            h, displacement = self.gru_cells(x[:, t], h, x[:, t - 1])
            outputs.append(h)
            displacements.append(displacement)
        if displacements:
            displacement_field = torch.stack(displacements, dim=1)
        else:
            displacement_field = x.new_zeros(B, 0, 2, H, W)
        return torch.stack(outputs, dim=1), displacement_field


class MotionEncoder(Encoder):
    """Shared U-Net encoder followed by one MA-GRU per scale."""

    def __init__(self, n_channels=1, gru_scales=(1, 2, 3, 4, 5)):
        super().__init__(n_channels)
        self.gru_scales = tuple(gru_scales)
        for i, channels in enumerate(FEATURES, start=1):
            setattr(self, f'dsa_gru{i}', MotionAwareGRU(channels) if i in self.gru_scales else None)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = super().forward(x.reshape(B * T, C, H, W))
        feats = [f.view(B, T, *f.shape[1:]).contiguous() for f in feats]
        out = {}
        displacement = None
        for i in range(5, 0, -1):
            gru = getattr(self, f'dsa_gru{i}')
            if gru is None:
                out[f'x{i}'] = feats[i - 1]
            else:
                out[f'x{i}'], disp = gru(feats[i - 1])
                if i == 1:
                    displacement = disp
        out['displacement'] = displacement
        return out


class MotionUNet(nn.Module):
    """MACOS network. Input: x (B, T, 1, H, W), the last frame being the key frame.

    forward(x) returns a dict with
        predictions:        (B, T, n_classes, H, W) logits, or a list of such tensors under deep supervision
        displacement_field: (B*(T-1), 2, H, W) finest-scale displacement, pair t being frame t -> frame t+1
        warped_image:       (B*(T-1), 1, H, W) frames 0..T-2 warped onto frames 1..T-1
        fixed_image:        (B*(T-1), 1, H, W) frames 1..T-1
    forward(x, key_frame_only=True) returns {'predictions': (B, n_classes, H, W)} for the key frame only."""

    def __init__(self, n_channels=1, n_classes=2, bilinear=False, deep_supervision=True, gru_scales=(1, 2, 3, 4, 5)):
        super().__init__()
        self.temporal_encoder = MotionEncoder(n_channels, gru_scales)
        self.temporal_decoder = TemporalDecoder(n_classes, bilinear, deep_supervision)

    def forward(self, x, key_frame_only=False):
        B, T, C, H, W = x.shape
        features = self.temporal_encoder(x)
        if key_frame_only:
            return {'predictions': self.temporal_decoder.forward_key_frame(features)}
        outputs = {'predictions': self.temporal_decoder.forward_sequence(features),
                   'displacement_field': None, 'warped_image': None, 'fixed_image': None}
        displacement = features['displacement']
        if displacement is not None and displacement.shape[1] > 0:
            assert displacement.shape[-2:] == (H, W), 'the finest-scale displacement must have the input resolution'
            displacement = displacement.reshape(B * (T - 1), 2, H, W)
            source = x[:, :-1].reshape(B * (T - 1), C, H, W)
            target = x[:, 1:].reshape(B * (T - 1), C, H, W)
            outputs['displacement_field'] = displacement
            outputs['warped_image'] = warp(source, displacement)
            outputs['fixed_image'] = target
        return outputs
