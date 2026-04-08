import glob
import os
import random

import torch
import torchvision


class ReplayBuffer:
    """
    Buffer of up to `max_size` previously generated images.
    push_and_pop randomly returns either a stored image or the fresh one,
    stabilising discriminator training.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    result.append(self.data[idx].clone())
                    self.data[idx] = element
                else:
                    result.append(element)
        return torch.cat(result, dim=0)


def denorm(tensor):
    """Convert [-1, 1] tensor back to [0, 1] for saving."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def save_images(epoch, real_P, fake_S, real_S, fake_P, save_dir='outputs'):
    """Save a 4-column grid: real_P | fake_S | real_S | fake_P."""
    os.makedirs(save_dir, exist_ok=True)
    grid = torchvision.utils.make_grid(
        torch.cat([denorm(real_P), denorm(fake_S),
                   denorm(real_S), denorm(fake_P)], dim=0),
        nrow=4,
    )
    torchvision.utils.save_image(grid, os.path.join(save_dir, f'epoch_{epoch:03d}.png'))


def save_checkpoint(epoch, G_P2S, G_S2P, D_P, D_S,
                    optimizer_G, optimizer_D_P, optimizer_D_S,
                    checkpoint_dir='checkpoints', use_pretrained=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pth')
    payload = {
        'epoch': epoch,
        'G_P2S': G_P2S.state_dict(),
        'G_S2P': G_S2P.state_dict(),
        'D_P':   D_P.state_dict(),
        'D_S':   D_S.state_dict(),
        'opt_G':   optimizer_G.state_dict(),
        'opt_D_P': optimizer_D_P.state_dict(),
        'opt_D_S': optimizer_D_S.state_dict(),
    }
    if use_pretrained is not None:
        payload['use_pretrained'] = use_pretrained
    torch.save(payload, path)
    print(f'  [✓] Checkpoint saved → {path}')


def find_latest_checkpoint(checkpoint_dir):
    """
    Scan checkpoint_dir for files matching epoch_NNN.pth.
    Return the path with the highest epoch number, or None if none exist.
    """
    pattern = os.path.join(checkpoint_dir, 'epoch_*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda f: int(os.path.basename(f)
                                   .replace('epoch_', '')
                                   .replace('.pth', '')))
    return files[-1]


def load_checkpoint(path, G_P2S, G_S2P, D_P, D_S,
                    optimizer_G, optimizer_D_P, optimizer_D_S, device):
    """
    Load a checkpoint saved by the training loop.
    Returns the epoch number to resume FROM (i.e. saved_epoch + 1).
    """
    print(f"[Resume] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    G_P2S.load_state_dict(ckpt['G_P2S'])
    G_S2P.load_state_dict(ckpt['G_S2P'])
    D_P.load_state_dict(ckpt['D_P'])
    D_S.load_state_dict(ckpt['D_S'])
    optimizer_G.load_state_dict(ckpt['opt_G'])
    optimizer_D_P.load_state_dict(ckpt['opt_D_P'])
    optimizer_D_S.load_state_dict(ckpt['opt_D_S'])

    saved_epoch = ckpt['epoch']
    print(f"[Resume] Restored from epoch {saved_epoch}. "
          f"Resuming from epoch {saved_epoch + 1}.")

    if 'use_pretrained' in ckpt:
        print(f"[Resume] Checkpoint was trained with "
              f"USE_PRETRAINED={ckpt['use_pretrained']}")

    return saved_epoch + 1
