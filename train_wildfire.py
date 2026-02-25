"""
Training entry point for SwinUnet on the WildfireSpreadTS dataset.

Key design:
  - Time steps are concatenated as channel groups: in_chans = n_leading_observations * 40
  - Binary segmentation: num_classes = 2  (0=no fire, 1=fire)
  - No pretrained weights (patch embed shape differs from ImageNet)

Example usage:
  python train_wildfire.py \
    --data_dir /path/to/wildfire_hdf5 \
    --output_dir ./runs/wildfire_t1 \
    --cfg configs/swin_tiny_patch4_window4_128_wildfire.yaml \
    --n_leading_observations 1 \
    --load_from_hdf5
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from networks.vision_transformer import SwinUnet
from trainer_wildfire import trainer_wildfire
from datasets.wildfire import N_FEATURES_PER_TIMESTEP

parser = argparse.ArgumentParser()

# --- Data ---
parser.add_argument('--data_dir', type=str, required=True,
                    help='Root directory of the WildfireSpreadTS HDF5 dataset')
parser.add_argument('--n_leading_observations', type=int, default=1,
                    help='Number of input timesteps (days). in_chans = n_leading_observations * 40')
parser.add_argument('--n_leading_observations_test_adjustment', type=int, default=None,
                    help='Align test-set size across different n_leading_observations values. '
                         'Defaults to n_leading_observations.')
parser.add_argument('--crop_side_length', type=int, default=128,
                    help='Side length of random square crops used during training')
parser.add_argument('--load_from_hdf5', action='store_true', default=True,
                    help='Load data from HDF5 files (recommended for speed)')
parser.add_argument('--data_fold_id', type=int, default=0, choices=range(12),
                    help='Which train/val/test year split to use (0–11)')

# --- Model / training ---
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory for checkpoints and logs')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--eval_interval', type=int, default=1,
                    help='Run validation every N epochs')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--focal_gamma', type=float, default=2.0,
                    help='Focal loss gamma (focusing parameter). Paper uses 2.0.')

# --- Swin config (passed through to yacs) ---
parser.add_argument('--cfg', type=str,
                    default='configs/swin_tiny_patch4_window4_128_wildfire.yaml',
                    metavar='FILE', help='Path to model config YAML')
parser.add_argument('--opts', nargs='+', default=None,
                    help='Override config options as KEY VALUE pairs, '
                         'e.g. MODEL.SWIN.EMBED_DIM 128')
parser.add_argument('--resume', default=None, help='Checkpoint to resume from')
parser.add_argument('--use-checkpoint', action='store_true',
                    help='Use gradient checkpointing to save GPU memory')
parser.add_argument('--amp-opt-level', type=str, default='O1',
                    choices=['O0', 'O1', 'O2'])
parser.add_argument('--tag', default=None)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--throughput', action='store_true')

# Kept for yacs config compat
parser.add_argument('--zip', action='store_true')
parser.add_argument('--cache-mode', type=str, default='part',
                    choices=['no', 'full', 'part'])
parser.add_argument('--accumulation-steps', type=int, default=None)

args = parser.parse_args()

# Inject in_chans into opts so yacs picks it up
in_chans = args.n_leading_observations * N_FEATURES_PER_TIMESTEP
extra_opts = [
    'MODEL.SWIN.IN_CHANS', str(in_chans),
    'MODEL.PRETRAIN_CKPT', 'None',   # no compatible pretrain
]
args.opts = (args.opts or []) + extra_opts

config = get_config(args)


if __name__ == '__main__':
    if not args.deterministic if hasattr(args, 'deterministic') else False:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.n_leading_observations_test_adjustment is None:
        args.n_leading_observations_test_adjustment = args.n_leading_observations

    os.makedirs(args.output_dir, exist_ok=True)

    net = SwinUnet(config, img_size=config.DATA.IMG_SIZE, num_classes=2).cuda()
    # No pretrained weights — patch embed channel count differs from ImageNet
    print(f"Model in_chans={in_chans}  (n_leading_observations={args.n_leading_observations} × 40 features)")
    print(f"Model parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")

    trainer_wildfire(args, net, args.output_dir)
