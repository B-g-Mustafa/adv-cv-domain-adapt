"""
Qualitative evaluation: load a checkpoint and generate translated images
for a fixed set of samples from each domain.

Usage:
    python evaluate.py --checkpoint checkpoints_amazon/spatial/epoch_199.pth \
                       --mode spatial --n_samples 16
    python evaluate.py --checkpoint checkpoints_amazon/spectral/epoch_199.pth \
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
    G_A2D = Generator().to(device)
    G_D2A = Generator().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    G_A2D.load_state_dict(ckpt['G_P2S'])
    G_D2A.load_state_dict(ckpt['G_S2P'])
    G_A2D.eval()
    G_D2A.eval()
    return G_A2D, G_D2A


def evaluate(checkpoint_path, mode='spatial', n_samples=16,
             save_dir='outputs/eval'):
    device = torch.device(config.DEVICE)
    G_A2D, G_D2A = load_generators(checkpoint_path, device)

    loader_A, loader_D = get_loaders()

    os.makedirs(save_dir, exist_ok=True)

    amazons = []
    webcams   = []
    for real_A in loader_A:
        amazons.append(real_A)
        if len(amazons) >= n_samples:
            break
    for real_D in loader_D:
        webcams.append(real_D)
        if len(webcams) >= n_samples:
            break

    amazons = torch.cat(amazons[:n_samples], dim=0).to(device)
    webcams   = torch.cat(webcams[:n_samples],   dim=0).to(device)

    with torch.no_grad():
        if mode == 'spatial':
            fake_D = G_A2D(amazons)
            fake_A = G_D2A(webcams)
        else:
            fake_D = spectral_translate(G_A2D, amazons, config.BETA_FREQ)
            fake_A = spectral_translate(G_D2A, webcams,   config.BETA_FREQ)

    # Save amazon → webcam grid
    grid_A2D = torchvision.utils.make_grid(
        torch.cat([denorm(amazons), denorm(fake_D)], dim=0), nrow=n_samples
    )
    torchvision.utils.save_image(grid_A2D, os.path.join(save_dir, 'amazon_to_webcam.png'))

    # Save webcam → amazon grid
    grid_D2A = torchvision.utils.make_grid(
        torch.cat([denorm(webcams), denorm(fake_A)], dim=0), nrow=n_samples
    )
    torchvision.utils.save_image(grid_D2A, os.path.join(save_dir, 'webcam_to_amazon.png'))

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
