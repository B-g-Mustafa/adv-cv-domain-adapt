"""
Pure Frequency Spectral CycleGAN training script.
The GAN architecture operates entirely on normalized low-frequency
amplitude maps. Spatial pixels are only reconstructed for visualization.
Amazon ↔ webcam domain adaptation on the Office-31 dataset.
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

def get_gaussian_mask(B, C, H, W, beta, device):
    """Creates a smooth Gaussian mask to prevent ringing artifacts."""
    mask = torch.zeros(B, C, H, W, device=device)
    cy, cx = H // 2, W // 2

    # Create coordinate grids
    y = torch.arange(0, H, device=device).float() - cy
    x = torch.arange(0, W, device=device).float() - cx
    y, x = torch.meshgrid(y, x, indexing='ij')

    # Calculate radius squared
    r2 = (x ** 2) + (y ** 2)

    # Variance based on beta
    sigma = (beta * H) / 2.0

    # Gaussian formula
    gaussian = torch.exp(-r2 / (2 * sigma ** 2))

    # Broadcast to match batch and channels
    mask[0, 0, :, :] = gaussian
    return mask.expand(B, C, H, W)


def encode_to_amplitude_map(img, beta):
    """
    Convert spatial image into a normalized low-freq amplitude map [0,1].
    Returns the map, plus all original data needed to reverse the process.
    """
    B, C, H, W = img.shape
    device = img.device

    fft = to_freq(img)

    # mask = get_low_mask(B, C, H, W, beta, device)
    mask = get_gaussian_mask(B, C, H, W, beta, device)

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
    loader_A, loader_D = get_loaders()
    print(f'Amazon images: {len(loader_A.dataset)}')
    print(f'webcam images  : {len(loader_D.dataset)}')

    # Initialize standard models. They don't need to know they are processing
    # frequencies instead of pixels; to them, it's just a 3-channel [0,1] tensor.
    G_A2D = Generator().to(device)
    G_D2A = Generator().to(device)
    D_A = Discriminator().to(device)
    D_D = Discriminator().to(device)

    init_weights(G_A2D)
    init_weights(G_D2A)
    init_weights(D_A)
    init_weights(D_D)

    if config.USE_PRETRAINED:
        load_pretrained_encoder(G_A2D)
        load_pretrained_encoder(G_D2A)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_A2D.parameters(), G_D2A.parameters()),
        lr=config.LR, betas=(config.BETA1, config.BETA2),
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    optimizer_D_D = torch.optim.Adam(D_D.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))

    lr_lambda = build_lr_lambda(config.NUM_EPOCHS)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda)
    scheduler_D_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D_D, lr_lambda)

    buffer_D = ReplayBuffer(config.BUFFER_SIZE)
    buffer_A = ReplayBuffer(config.BUFFER_SIZE)

    # I recommend starting with 0.05 for pure frequency mapping
    beta = getattr(config, 'BETA_FREQ_PURE', 0.05)

    # ---------------------------------------------------------- resume logic
    start_epoch = 0

    if config.RESUME:
        if config.RESUME_EPOCH == -1:
            ckpt_path = find_latest_checkpoint(config.CHECKPOINT_DIR + '/pure_spectral')
        else:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR + '/pure_spectral',
                f'epoch_{config.RESUME_EPOCH:03d}.pth',
            )

        if ckpt_path and os.path.exists(ckpt_path):
            start_epoch = load_checkpoint(
                ckpt_path,
                G_A2D, G_D2A, D_A, D_D,
                optimizer_G, optimizer_D_A, optimizer_D_D,
                device,
            )
        else:
            print(f"[Resume] No checkpoint found at: {ckpt_path}")
            print("[Resume] Starting from scratch instead.")
    else:
        print("[Scratch] RESUME=False — starting from epoch 0.")

    # Fast-forward LR schedulers to the correct position
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler_G.step()
            scheduler_D_A.step()
            scheduler_D_D.step()
        print(f"[Resume] LR schedulers fast-forwarded to epoch {start_epoch}.")

    # ------------------------------------------ startup banner (post-resume)
    print("=" * 50)
    print(f"  PURE FREQUENCY CycleGAN — Office-31 Amazon → webcam")
    print(f"  Beta Window        : {beta}")
    print(f"  Pretrained encoder : {config.USE_PRETRAINED}")
    print(f"  Device             : {config.DEVICE}")
    print(f"  Epochs             : {config.NUM_EPOCHS}")
    print(f"  Resume             : {config.RESUME}")
    if config.RESUME:
        mode = 'latest' if config.RESUME_EPOCH == -1 else f'epoch {config.RESUME_EPOCH}'
        print(f"  Resume mode        : {mode}")
    print(f"  Starting epoch     : {start_epoch}")
    print("=" * 50)

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        for batch_idx, (real_A_img, real_D_img) in enumerate(zip(loader_A, loader_D)):

            real_A_img = real_A_img.to(device)
            real_D_img = real_D_img.to(device)

            # ================== PRE-PROCESSING TO AMPLITUDE MAPS ==================
            # Transform spatial images into GAN-ready amplitude maps
            amp_A, phase_A, vmin_A, vmax_A, mask_A, fft_A = encode_to_amplitude_map(real_A_img, beta)
            amp_D, phase_D, vmin_D, vmax_D, mask_D, fft_D = encode_to_amplitude_map(real_D_img, beta)

            # ====================== Train Generators ======================
            optimizer_G.zero_grad()

            # Identity loss (using amplitude maps)
            id_amp_A = G_D2A(amp_A)
            id_amp_D = G_A2D(amp_D)
            loss_id_A = criterion_identity(id_amp_A, amp_A)
            loss_id_D = criterion_identity(id_amp_D, amp_D)

            # Forward translations
            fake_amp_D = G_A2D(amp_A)
            fake_amp_A = G_D2A(amp_D)

            # Adversarial losses
            loss_adv_A2D = criterion_GAN(D_D(fake_amp_D), torch.ones_like(D_D(fake_amp_D)))
            loss_adv_D2A = criterion_GAN(D_A(fake_amp_A), torch.ones_like(D_A(fake_amp_A)))

            # Cycle consistency losses
            rec_amp_A = G_D2A(fake_amp_D)
            rec_amp_D = G_A2D(fake_amp_A)
            loss_cycle_A = criterion_cycle(rec_amp_A, amp_A)
            loss_cycle_D = criterion_cycle(rec_amp_D, amp_D)

            loss_G = (
                    loss_adv_A2D + loss_adv_D2A
                    + config.LAMBDA_CYCLE * (loss_cycle_A + loss_cycle_D)
                    + config.LAMBDA_IDENTITY * (loss_id_A + loss_id_D)
            )
            loss_G.backward()
            optimizer_G.step()

            # ================== Train Discriminator D_D ===================
            optimizer_D_D.zero_grad()
            fake_amp_D_buf = buffer_D.push_and_pop(fake_amp_D.detach())
            pred_real_D = D_D(amp_D)
            pred_fake_D = D_D(fake_amp_D_buf)
            loss_D_D = (criterion_GAN(pred_real_D, torch.ones_like(pred_real_D)) +
                        criterion_GAN(pred_fake_D, torch.zeros_like(pred_fake_D))) * 0.5
            loss_D_D.backward()
            optimizer_D_D.step()

            # ================== Train Discriminator D_A ===================
            optimizer_D_A.zero_grad()
            fake_amp_A_buf = buffer_A.push_and_pop(fake_amp_A.detach())
            pred_real_A = D_A(amp_A)
            pred_fake_A = D_A(fake_amp_A_buf)
            loss_D_A = (criterion_GAN(pred_real_A, torch.ones_like(pred_real_A)) +
                        criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch:3d}/{config.NUM_EPOCHS}] '
                      f'Batch [{batch_idx:4d}] '
                      f'loss_G={loss_G.item():.4f} '
                      f'loss_D_A={loss_D_A.item():.4f} '
                      f'loss_D_D={loss_D_D.item():.4f}')

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_D.step()

        # ====================== RECONSTRUCT IMAGES FOR SAVING ======================
        if epoch % config.SAVE_EVERY == 0:
            with torch.no_grad():
                # We use decode_to_spatial_image to translate the GAN's amplitude map back into pixels
                fake_D_img = decode_to_spatial_image(fake_amp_D, phase_A, vmin_A, vmax_A, mask_A, fft_A)
                fake_A_img = decode_to_spatial_image(fake_amp_A, phase_D, vmin_D, vmax_D, mask_D, fft_D)

                # Save out the spatial pixels so you can evaluate the visual quality
                save_images(epoch, real_A_img, fake_D_img, real_D_img, fake_A_img,
                            save_dir=config.OUTPUT_DIR + '/pure_spectral')
        # ====================== SAVE CHECKPOINTS ======================
        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            save_checkpoint(
                epoch, G_A2D, G_D2A, D_A, D_D,
                optimizer_G, optimizer_D_A, optimizer_D_D,
                checkpoint_dir=config.CHECKPOINT_DIR + '/pure_spectral',
                use_pretrained=config.USE_PRETRAINED,
            )

    print('Training complete.')


if __name__ == '__main__':
    train()
