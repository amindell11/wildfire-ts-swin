"""
SimMIM-style self-supervised pre-training wrapper for SwinTransformerSys.

Uses channel-group masking: 5 semantic groups of channels are independently
masked per spatial patch. The model must reconstruct masked channel-groups
from the visible ones, learning cross-feature spatial relationships.

Channel groups (post-preprocessing, 40 channels):
  Group 0 - Spectral/Veg  [5ch]:  0-4   (M11, I2, I1, NDVI, EVI2)
  Group 1 - Weather       [7ch]:  5-11  (precip, wind, temp, ERC, humidity)
  Group 2 - Terrain       [3ch]:  12-14 (slope, aspect, elevation)
  Group 3 - Land+Drought [18ch]:  15-32 (PDSI, 17x landcover one-hot)
  Group 4 - Forecast+AF   [7ch]:  33-39 (5x forecast, AF, binary_AF)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

# Channel indices for each semantic group (post-preprocessing 40-channel input)
CHANNEL_GROUPS = [
    list(range(0, 5)),     # spectral / vegetation
    list(range(5, 12)),    # weather
    list(range(12, 15)),   # terrain
    list(range(15, 33)),   # landcover + drought
    list(range(33, 40)),   # forecast + active fire
]
N_GROUPS = len(CHANNEL_GROUPS)


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
        """Generate a full-image channel-group mask.

        Each of the 5 channel groups is independently masked across the
        entire image with probability self.mask_prob. At least 1 group
        must be masked and at least 1 visible per sample.

        This forces the model to learn cross-feature relationships rather
        than spatially interpolating from neighboring patches.

        Returns:
            mask_pixels: (B, 40, H, W) float tensor — 1 = masked, 0 = visible
        """
        H = self.patches_h * self.patch_size
        W = self.patches_w * self.patch_size

        # (B, N_GROUPS) Bernoulli mask — entire group on or off per sample
        group_mask = torch.rand(B, N_GROUPS, device=device) < self.mask_prob

        # Ensure at least 1 masked and 1 visible per sample
        for b in range(B):
            if group_mask[b].all():
                # All masked — unmask one random group
                group_mask[b, torch.randint(N_GROUPS, (1,), device=device)] = False
            if (~group_mask[b]).all():
                # All visible — mask one random group
                group_mask[b, torch.randint(N_GROUPS, (1,), device=device)] = True

        # Expand groups to channels: (B, N_GROUPS) → (B, 40, H, W)
        channel_mask = torch.zeros(B, self.in_chans, device=device)
        for g, ch_indices in enumerate(CHANNEL_GROUPS):
            channel_mask[:, ch_indices] = group_mask[:, g:g+1].float()

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
