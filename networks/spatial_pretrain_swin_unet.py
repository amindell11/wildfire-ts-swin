"""
Spatial inpainting pre-training wrapper for SwinTransformerSys.

Masks a large contiguous block (e.g. 64x64 pixels = ~24km) across ALL
channels simultaneously. The model must reconstruct the missing region
from the surrounding spatial context.

This can't be solved by:
  - Copying (the region is completely gone)
  - Neighbor interpolation (24km gap is too large)
  - Cross-feature shortcuts (all channels masked together)

The model must learn actual spatial structure: how terrain shapes weather,
how vegetation follows elevation gradients, how weather fields have
coherent spatial patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SpatialInpaintSwinUnet(nn.Module):
    def __init__(
        self,
        config,
        img_size=128,
        use_factored_embed=False,
        in_chans=40,
        mask_size=64,
    ):
        """
        Args:
            config: yacs CfgNode
            img_size: spatial input size
            use_factored_embed: whether to use FactoredPatchEmbed
            in_chans: number of input channels (40 after preprocessing)
            mask_size: side length of the square masked region in pixels
        """
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.mask_size = mask_size

        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
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
        """Generate a random spatial block mask.

        For each sample, places a mask_size x mask_size block at a random
        position. All 40 channels are masked in that region.

        Returns:
            mask: (B, 1, H, W) float tensor — 1 = masked, 0 = visible
                  Broadcast across channels since all channels are masked together.
        """
        H, W = self.img_size, self.img_size
        mask = torch.zeros(B, 1, H, W, device=device)

        for b in range(B):
            top = torch.randint(0, H - self.mask_size + 1, (1,)).item()
            left = torch.randint(0, W - self.mask_size + 1, (1,)).item()
            mask[b, :, top:top + self.mask_size, left:left + self.mask_size] = 1.0

        return mask

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, 40, H, W) preprocessed input
            mask: optional precomputed mask (B, 1, H, W); generated if None

        Returns:
            loss: scalar L1 reconstruction loss on masked region only
            pred: (B, 40, H, W) full reconstruction
            mask: (B, 1, H, W) the mask used
        """
        B = x.shape[0]
        target = x.clone()

        if mask is None:
            mask = self.generate_mask(B, x.device)

        # Zero out the masked spatial region across all channels
        x_masked = x * (1.0 - mask)

        # Full encoder + decoder forward
        pred = self.swin_unet(x_masked)  # (B, 40, H, W)

        # L1 loss on masked region only (mask broadcasts across channels)
        loss = (F.l1_loss(pred, target, reduction='none') * mask).sum()
        loss = loss / (mask.sum() * self.in_chans + 1e-6)

        return loss, pred, mask

    def get_encoder_decoder_state_dict(self):
        """Return transferable weights (excludes output head)."""
        return {
            k: v for k, v in self.swin_unet.state_dict().items()
            if 'output' not in k
        }
