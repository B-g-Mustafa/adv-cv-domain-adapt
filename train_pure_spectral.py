"""
Pure Frequency Spectral CycleGAN training script.
The GAN architecture operates entirely on normalized low-frequency
amplitude maps. Spatial pixels are only reconstructed for visualization.
"""

import itertools
import os

import torch
import torch.nn as nn

import config
from dataset import get_loaders
from models import Generator, Discriminator, init_weights, load_pretrained_encoder
from utils import ReplayBuffer, save_images, save_checkpoint, find_latest_checkpoint, load_checkpoint


# ---------------------------------------------------------------- FFT Utilities

def to_freq(img):
    """Convert spatial image to centered complex FFT."""
    return torch.fft.fftshift(torch.fft.fft2(img.float()))


def from_freq(fft):
    """Convert centered complex FFT back to spatial image."""
    return torch.fft.ifft2(torch.fft.ifftshift(fft)).real


def get_low_mask(B, C, H, W, beta, device):
    """Create a boolean mask for the low-frequency center."""
    mask = torch.zeros(B, C, H, W, device=device)
    cy, cx = H // 2, W // 2
    bh = max(1, int(beta * H))
    bw = max(1, int(beta * W))
    mask[:, :, cy - bh:cy + bh, cx - bw:cx + bw] = 1.0
    return mask


def encode_to_amplitude_map(img, beta):
    """
    Convert spatial image into a normalized low-freq amplitude map [0,1].
    Returns the map, plus all original data needed to reverse the process.
    """
    B, C, H, W = img.shape
    device = img.device

    fft = to_freq(img)
    mask = get_low_mask(B, C, H, W, beta, device)

    # 1. Extract Phase (crucial for structural reconstruction later)
    phase = torch.angle(fft)

    # 2. Extract Amplitude and apply mask
    amp = torch.abs(fft)
    masked_amp = amp * mask

    # 3. Log-scale the amplitude to compress massive FFT values
    amp_log = torch.log1p(masked_amp)

    # 4. Min-Max Normalization to strictly [0, 1] bounds for the GAN
    # We flatten the spatial dims to find min/max per channel per image
    amp_flat = amp_log.view(B, C, -1)
    vmin = amp_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    vmax = amp_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)

    denom = vmax - vmin
    denom[denom == 0] = 1e-8  # Prevent division by zero
    amp_norm = (amp_log - vmin) / denom

    # We return amp_norm for the GAN, and the rest for the reconstruction phase
    return amp_norm, phase, vmin, vmax, mask, fft


def decode_to_spatial_image(fake_amp_norm, original_phase, vmin, vmax, mask, original_fft):
    """
    Take the GAN's output amplitude map and mathematically reconstruct
    the final spatial image using the original source's phase and high frequencies.
    """
    # 1. Reverse Min-Max Normalization
    amp_log = fake_amp_norm * (vmax - vmin) + vmin

    # 2. Reverse Log-Scale (expm1 is the inverse of log1p)
    fake_amp = torch.expm1(amp_log)

    # 3. Combine new fake amplitude with the original phase
    fake_low_fft = fake_amp * torch.exp(1j * original_phase)

    # 4. Extract original high frequencies (the pencil lines/sharp edges)
    original_high_fft = original_fft * (1.0 - mask)

    # 5. Blend: Fake Low Freqs + Original High Freqs
    new_fft = (fake_low_fft * mask) + original_high_fft

    # 6. Inverse FFT to pixel space
    img_out = from_freq(new_fft)

    # Clamp to [-1, 1] just in case of minor floating point artifacts
    return torch.clamp(img_out, -1.0, 1.0)


# ---------------------------------------------------------------- lr schedule
def build_lr_lambda(num_epochs=config.NUM_EPOCHS):
    decay_start = num_epochs // 2

    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / float(decay_start))

    return lr_lambda


# -------------------------------------------------------------------- training
def train():
    device = torch.device(config.DEVICE)
    loader_photo, loader_sketch = get_loaders()
    print(f'Photo images : {len(loader_photo.dataset)}')
    print(f'Sketch images: {len(loader_sketch.dataset)}')

    # Initialize standard models. They don't need to know they are processing
    # frequencies instead of pixels; to them, it's just a 3-channel [0,1] tensor.
    G_P2S = Generator().to(device)
    G_S2P = Generator().to(device)
    D_P = Discriminator().to(device)
    D_S = Discriminator().to(device)

    init_weights(G_P2S)
    init_weights(G_S2P)
    init_weights(D_P)
    init_weights(D_S)

    if config.USE_PRETRAINED:
        load_pretrained_encoder(G_P2S)
        load_pretrained_encoder(G_S2P)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_P2S.parameters(), G_S2P.parameters()),
        lr=config.LR, betas=(config.BETA1, config.BETA2),
    )
    optimizer_D_P = torch.optim.Adam(D_P.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    optimizer_D_S = torch.optim.Adam(D_S.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))

    lr_lambda = build_lr_lambda(config.NUM_EPOCHS)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D_P = torch.optim.lr_scheduler.LambdaLR(optimizer_D_P, lr_lambda)
    scheduler_D_S = torch.optim.lr_scheduler.LambdaLR(optimizer_D_S, lr_lambda)

    buffer_S = ReplayBuffer(config.BUFFER_SIZE)
    buffer_P = ReplayBuffer(config.BUFFER_SIZE)

    # I recommend starting with 0.05 for pure frequency mapping
    beta = getattr(config, 'BETA_FREQ', 0.05)

    print("=" * 50)
    print(f"  PURE FREQUENCY CycleGAN — PACS Photo -> Sketch")
    print(f"  Beta Window        : {beta}")
    print("=" * 50)

    for epoch in range(config.NUM_EPOCHS):
        for batch_idx, (real_P_img, real_S_img) in enumerate(zip(loader_photo, loader_sketch)):

            real_P_img = real_P_img.to(device)
            real_S_img = real_S_img.to(device)

            # ================== PRE-PROCESSING TO AMPLITUDE MAPS ==================
            # Transform spatial images into GAN-ready amplitude maps
            amp_P, phase_P, vmin_P, vmax_P, mask_P, fft_P = encode_to_amplitude_map(real_P_img, beta)
            amp_S, phase_S, vmin_S, vmax_S, mask_S, fft_S = encode_to_amplitude_map(real_S_img, beta)

            # ====================== Train Generators ======================
            optimizer_G.zero_grad()

            # Identity loss (using amplitude maps)
            id_amp_P = G_S2P(amp_P)
            id_amp_S = G_P2S(amp_S)
            loss_id_P = criterion_identity(id_amp_P, amp_P)
            loss_id_S = criterion_identity(id_amp_S, amp_S)

            # Forward translations
            fake_amp_S = G_P2S(amp_P)
            fake_amp_P = G_S2P(amp_S)

            # Adversarial losses
            loss_adv_P2S = criterion_GAN(D_S(fake_amp_S), torch.ones_like(D_S(fake_amp_S)))
            loss_adv_S2P = criterion_GAN(D_P(fake_amp_P), torch.ones_like(D_P(fake_amp_P)))

            # Cycle consistency losses
            rec_amp_P = G_S2P(fake_amp_S)
            rec_amp_S = G_P2S(fake_amp_P)
            loss_cycle_P = criterion_cycle(rec_amp_P, amp_P)
            loss_cycle_S = criterion_cycle(rec_amp_S, amp_S)

            loss_G = (
                    loss_adv_P2S + loss_adv_S2P
                    + config.LAMBDA_CYCLE * (loss_cycle_P + loss_cycle_S)
                    + config.LAMBDA_IDENTITY * (loss_id_P + loss_id_S)
            )
            loss_G.backward()
            optimizer_G.step()

            # ================== Train Discriminator D_S ===================
            optimizer_D_S.zero_grad()
            fake_amp_S_buf = buffer_S.push_and_pop(fake_amp_S.detach())
            pred_real_S = D_S(amp_S)
            pred_fake_S = D_S(fake_amp_S_buf)
            loss_D_S = (criterion_GAN(pred_real_S, torch.ones_like(pred_real_S)) +
                        criterion_GAN(pred_fake_S, torch.zeros_like(pred_fake_S))) * 0.5
            loss_D_S.backward()
            optimizer_D_S.step()

            # ================== Train Discriminator D_P ===================
            optimizer_D_P.zero_grad()
            fake_amp_P_buf = buffer_P.push_and_pop(fake_amp_P.detach())
            pred_real_P = D_P(amp_P)
            pred_fake_P = D_P(fake_amp_P_buf)
            loss_D_P = (criterion_GAN(pred_real_P, torch.ones_like(pred_real_P)) +
                        criterion_GAN(pred_fake_P, torch.zeros_like(pred_fake_P))) * 0.5
            loss_D_P.backward()
            optimizer_D_P.step()

            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch:3d}/{config.NUM_EPOCHS}] '
                      f'Batch [{batch_idx:4d}] '
                      f'loss_G={loss_G.item():.4f} '
                      f'loss_D_P={loss_D_P.item():.4f} '
                      f'loss_D_S={loss_D_S.item():.4f}')

        scheduler_G.step()
        scheduler_D_P.step()
        scheduler_D_S.step()

        # ====================== RECONSTRUCT IMAGES FOR SAVING ======================
        if epoch % config.SAVE_EVERY == 0:
            with torch.no_grad():
                # We use decode_to_spatial_image to translate the GAN's amplitude map back into pixels
                fake_S_img = decode_to_spatial_image(fake_amp_S, phase_P, vmin_P, vmax_P, mask_P, fft_P)
                fake_P_img = decode_to_spatial_image(fake_amp_P, phase_S, vmin_S, vmax_S, mask_S, fft_S)

                # Save out the spatial pixels so you can evaluate the visual quality
                save_images(epoch, real_P_img, fake_S_img, real_S_img, fake_P_img,
                            save_dir=config.OUTPUT_DIR + '/pure_spectral')

    print('Training complete.')


if __name__ == '__main__':
    train()