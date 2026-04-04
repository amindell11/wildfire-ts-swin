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
        """Generate a per-channel-group, per-patch binary mask.

        For each patch, each of the 5 channel groups is independently masked
        with probability self.mask_prob. At least 1 group is masked and at
        least 1 is visible (resampled if all-same).

        Returns:
            mask_pixels: (B, 40, H, W) float tensor — 1 = masked, 0 = visible
        """
        pH, pW = self.patches_h, self.patches_w

        # (B, N_GROUPS, pH * pW) Bernoulli mask
        group_mask = torch.rand(B, N_GROUPS, pH * pW, device=device) < self.mask_prob

        # Ensure at least 1 masked and 1 visible per patch
        all_masked = group_mask.all(dim=1)    # (B, pH*pW)
        all_visible = (~group_mask).all(dim=1)
        for b in range(B):
            # Fix patches where all groups are masked: unmask one random group
            bad = all_masked[b].nonzero(as_tuple=True)[0]
            if len(bad) > 0:
                rand_groups = torch.randint(N_GROUPS, (len(bad),), device=device)
                group_mask[b, rand_groups, bad] = False
            # Fix patches where all groups are visible: mask one random group
            bad = all_visible[b].nonzero(as_tuple=True)[0]
            if len(bad) > 0:
                rand_groups = torch.randint(N_GROUPS, (len(bad),), device=device)
                group_mask[b, rand_groups, bad] = True

        # Expand groups to channels: (B, N_GROUPS, pH*pW) → (B, 40, pH, pW)
        channel_mask = torch.zeros(B, self.in_chans, pH * pW, device=device)
        for g, ch_indices in enumerate(CHANNEL_GROUPS):
            channel_mask[:, ch_indices, :] = group_mask[:, g:g+1, :].float()

        channel_mask = channel_mask.reshape(B, self.in_chans, pH, pW)

        # Upscale to pixel space: (B, 40, H, W)
        mask_pixels = channel_mask.repeat_interleave(self.patch_size, dim=2) \
                                  .repeat_interleave(self.patch_size, dim=3)
        return mask_pixels

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
