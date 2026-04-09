"""
Qualitative evaluation: load a checkpoint and generate translated images
for a fixed set of samples from each domain.

Usage:
    python evaluate.py --checkpoint checkpoints/spatial/ckpt_epoch_199.pth \
                       --mode spatial --n_samples 16
    python evaluate.py --checkpoint checkpoints/spectral/ckpt_epoch_199.pth \
                       --mode spectral --n_samples 16
"""

import argparse
import os

import torch
import torchvision

import config
from dataset import get_loaders
from models import Generator
from utils import denorm
from train_spectral_bottleneck import spectral_translate


def load_generators(checkpoint_path, device):
    G_P2S = Generator().to(device)
    G_S2P = Generator().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    G_P2S.load_state_dict(ckpt['G_P2S'])
    G_S2P.load_state_dict(ckpt['G_S2P'])
    G_P2S.eval()
    G_S2P.eval()
    return G_P2S, G_S2P


def translate(G_P2S, G_S2P, img, mode, ref=None, beta=config.BETA_FREQ):
    """Run forward translation in either spatial or spectral mode."""
    if mode == 'spatial':
        fake_S = G_P2S(img)
        fake_P = G_S2P(img)
    else:
        fake_S = spectral_translate(G_P2S, img, ref, beta)
        fake_P = spectral_translate(G_S2P, img, img, beta)
    return fake_S, fake_P


def evaluate(checkpoint_path, mode='spatial', n_samples=16,
             save_dir='outputs/eval'):
    device = torch.device(config.DEVICE)
    G_P2S, G_S2P = load_generators(checkpoint_path, device)

    loader_photo, loader_sketch = get_loaders()

    os.makedirs(save_dir, exist_ok=True)

    photos  = []
    sketches = []
    for real_P in loader_photo:
        photos.append(real_P)
        if len(photos) >= n_samples:
            break
    for real_S in loader_sketch:
        sketches.append(real_S)
        if len(sketches) >= n_samples:
            break

    photos   = torch.cat(photos[:n_samples], dim=0).to(device)
    sketches = torch.cat(sketches[:n_samples], dim=0).to(device)

    with torch.no_grad():
        if mode == 'spatial':
            fake_S = G_P2S(photos)
            fake_P = G_S2P(sketches)
        else:
            # Use first sketch batch as reference style
            fake_S = spectral_translate(G_P2S, photos,   sketches, config.BETA_FREQ)
            fake_P = spectral_translate(G_S2P, sketches, photos,   config.BETA_FREQ)

    # Save photo → sketch grid
    grid_P2S = torchvision.utils.make_grid(
        torch.cat([denorm(photos), denorm(fake_S)], dim=0), nrow=n_samples
    )
    torchvision.utils.save_image(grid_P2S, os.path.join(save_dir, 'photo_to_sketch.png'))

    # Save sketch → photo grid
    grid_S2P = torchvision.utils.make_grid(
        torch.cat([denorm(sketches), denorm(fake_P)], dim=0), nrow=n_samples
    )
    torchvision.utils.save_image(grid_S2P, os.path.join(save_dir, 'sketch_to_photo.png'))

    print(f'Saved evaluation grids to {save_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pth checkpoint file')
    parser.add_argument('--mode', choices=['spatial', 'spectral'],
                        default='spatial')
    parser.add_argument('--n_samples', type=int, default=16)
    parser.add_argument('--save_dir', default='outputs/eval')
    args = parser.parse_args()

    evaluate(args.checkpoint, mode=args.mode,
             n_samples=args.n_samples, save_dir=args.save_dir)
