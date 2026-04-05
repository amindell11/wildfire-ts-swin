"""
Next-day prediction pre-training wrapper for SwinTransformerSys.

Input: today's 40-channel features (B, 40, H, W)
Output: predicted tomorrow's dynamic features (B, N_dynamic, H, W)
Loss: L1 on dynamic channels only (weather, vegetation, forecasts)

Static features (terrain, landcover) don't change day-to-day, so
predicting them would just reward copying. The model must learn
spatial weather/vegetation dynamics — how patterns evolve across
terrain — which is the same spatial propagation skill needed for
fire spread prediction.

Dynamic channels (post-preprocessing):
  0-4:   spectral/vegetation (M11, I2, I1, NDVI, EVI2)
  5-11:  weather (precip, wind spd/dir, temp min/max, ERC, humidity)
  33-37: forecasts (precip, wind spd/dir, temp, humidity)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

# Dynamic feature indices in the 40-channel input (change day-to-day)
DYNAMIC_CHANNELS = list(range(0, 12)) + list(range(33, 38))
# Indices: 0-11 (spectral, veg, weather) + 33-37 (forecasts)
# Excludes: 12-14 (terrain), 15 (PDSI ~static), 16-32 (landcover), 38-39 (AF = zeros)
N_DYNAMIC = len(DYNAMIC_CHANNELS)  # 17


class NextDaySwinUnet(nn.Module):
    def __init__(
        self,
        config,
        img_size=128,
        use_factored_embed=False,
        in_chans=40,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=in_chans,
            n_timesteps=getattr(config.MODEL.SWIN, 'N_TIMESTEPS', 1),
            use_factored_embed=use_factored_embed,
            num_classes=N_DYNAMIC,  # predict only dynamic channels
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

    def forward(self, x, target=None):
        """
        Args:
            x: (B, 40, H, W) today's features
            target: (B, 40, H, W) tomorrow's full features (None during inference)

        Returns:
            If target provided: (loss, pred) where pred is (B, N_DYNAMIC, H, W)
            If no target: pred
        """
        pred = self.swin_unet(x)  # (B, N_DYNAMIC, H, W)

        if target is not None:
            # Extract dynamic channels from target for loss
            target_dynamic = target[:, DYNAMIC_CHANNELS, ...]  # (B, N_DYNAMIC, H, W)
            loss = F.l1_loss(pred, target_dynamic)
            return loss, pred
        return pred

    def get_encoder_decoder_state_dict(self):
        """Return transferable weights (excludes output head)."""
        return {
            k: v for k, v in self.swin_unet.state_dict().items()
            if 'output' not in k
        }
