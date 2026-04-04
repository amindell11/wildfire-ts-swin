"""
SimMIM-style self-supervised pre-training wrapper for SwinTransformerSys.

Uses per-feature channel masking: each feature is independently masked across
the entire image. The model must reconstruct masked features from the visible
ones, learning cross-feature spatial relationships.

Landcover (17 one-hot channels) is treated as a single maskable unit since
the channels encode one categorical variable.

Maskable units (24 total, mapping to 40 post-preprocessing channels):
  0:  M11 (ch 0)          12: Slope (ch 12)
  1:  I2 (ch 1)           13: Aspect (ch 13)
  2:  I1 (ch 2)           14: Elevation (ch 14)
  3:  NDVI (ch 3)         15: PDSI (ch 15)
  4:  EVI2 (ch 4)         16: Landcover 17x one-hot (ch 16-32)
  5:  Precipitation (ch 5) 17: Fcst precip (ch 33)
  6:  Wind speed (ch 6)   18: Fcst wind speed (ch 34)
  7:  Wind direction (ch 7) 19: Fcst wind dir (ch 35)
  8:  Temp min (ch 8)     20: Fcst temperature (ch 36)
  9:  Temp max (ch 9)     21: Fcst humidity (ch 37)
  10: ERC (ch 10)         22: Active fire (ch 38)
  11: Humidity (ch 11)    23: Binary AF (ch 39)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

# Each maskable unit maps to one or more channel indices in the 40-channel input.
# Landcover (17 one-hot) is one unit; everything else is a single channel.
MASK_UNITS = (
    [[i] for i in range(16)]           # ch 0-15: individual features
    + [list(range(16, 33))]            # ch 16-32: landcover one-hot (1 unit)
    + [[i] for i in range(33, 40)]     # ch 33-39: individual features
)
N_MASK_UNITS = len(MASK_UNITS)  # 24

# Feature names for visualization / logging
MASK_UNIT_NAMES = [
    'M11', 'I2', 'I1', 'NDVI', 'EVI2',
    'Precip', 'Wind Spd', 'Wind Dir', 'Temp Min', 'Temp Max', 'ERC', 'Humidity',
    'Slope', 'Aspect', 'Elevation', 'PDSI',
    'Landcover',
    'Fcst Precip', 'Fcst Wind Spd', 'Fcst Wind Dir', 'Fcst Temp', 'Fcst Humidity',
    'Active Fire', 'Binary AF',
]


class SimMIMSwinUnet(nn.Module):
    def __init__(
        self,
        config,
        img_size=128,
        mask_prob=0.5,
        use_factored_embed=False,
        in_chans=40,
    ):
        """
        Args:
            config: yacs CfgNode (same as SwinUnet uses).
            img_size: spatial input size.
            mask_prob: per-group probability of masking at each patch.
            use_factored_embed: whether to use FactoredPatchEmbed.
            in_chans: number of input channels (40 after preprocessing).
        """
        super().__init__()
        self.mask_prob = mask_prob
        self.in_chans = in_chans
        self.patch_size = config.MODEL.SWIN.PATCH_SIZE
        self.patches_h = img_size // self.patch_size
        self.patches_w = img_size // self.patch_size
        self.num_patches = self.patches_h * self.patches_w

        # Backbone with num_classes = in_chans for pixel reconstruction
        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            n_timesteps=getattr(config.MODEL.SWIN, 'N_TIMESTEPS', 1),
            use_factored_embed=use_factored_embed,
            num_classes=in_chans,  # reconstruct all 40 channels
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )

    def generate_mask(self, B, device):
        """Generate a per-feature full-image mask.

        Each of the 24 maskable units (individual features, except landcover
        which is one unit of 17 one-hot channels) is independently masked
        across the entire image with probability self.mask_prob. At least 1
        unit must be masked and at least 1 visible per sample.

        Returns:
            mask_pixels: (B, 40, H, W) float tensor — 1 = masked, 0 = visible
        """
        H = self.patches_h * self.patch_size
        W = self.patches_w * self.patch_size

        # (B, N_MASK_UNITS) Bernoulli mask — per feature
        unit_mask = torch.rand(B, N_MASK_UNITS, device=device) < self.mask_prob

        # Ensure at least 1 masked and 1 visible per sample
        for b in range(B):
            if unit_mask[b].all():
                unit_mask[b, torch.randint(N_MASK_UNITS, (1,), device=device)] = False
            if (~unit_mask[b]).all():
                unit_mask[b, torch.randint(N_MASK_UNITS, (1,), device=device)] = True

        # Expand units to channels: (B, N_MASK_UNITS) → (B, 40)
        channel_mask = torch.zeros(B, self.in_chans, device=device)
        for u, ch_indices in enumerate(MASK_UNITS):
            channel_mask[:, ch_indices] = unit_mask[:, u:u+1].float()

        # Broadcast spatially
        mask_pixels = channel_mask[:, :, None, None].expand(B, self.in_chans, H, W)
        return mask_pixels.contiguous()

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, 40, H, W) preprocessed input
            mask: optional precomputed mask (B, 40, H, W); generated if None

        Returns:
            loss: scalar L1 reconstruction loss on masked positions
            pred: (B, 40, H, W) reconstruction
            mask: (B, 40, H, W) the mask used (1 = masked)
        """
        B = x.shape[0]
        target = x.clone()

        if mask is None:
            mask = self.generate_mask(B, x.device)

        # Zero out masked channels at masked patches
        x_masked = x * (1.0 - mask)

        # Full encoder + decoder forward
        pred = self.swin_unet(x_masked)  # (B, 40, H, W)

        # L1 loss on masked positions only
        loss = (F.l1_loss(pred, target, reduction='none') * mask).sum()
        loss = loss / (mask.sum() + 1e-6)

        return loss, pred, mask

    def get_encoder_decoder_state_dict(self):
        """Return state dict with only the transferable weights.

        Excludes the output Conv2d (40-channel reconstruction head) since
        fine-tuning uses num_classes=2.
        """
        return {
            k: v for k, v in self.swin_unet.state_dict().items()
            if 'output' not in k
        }
