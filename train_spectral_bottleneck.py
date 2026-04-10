"""
Spectral CycleGAN training script.
Identical to Spatial CycleGAN except generators operate on low-frequency
amplitude components of the FFT spectrum rather than the full pixel image.
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




def spectral_translate(G, src_img, beta):
    """
    Translate the low-frequency amplitude of src_img using generator G.
    Phase is perfectly preserved.

    Args:
        G       : Generator network
        src_img : source domain image (B, C, H, W)
        beta    : fraction of low frequencies to modify (0.0 to 1.0)

    Returns:
        img_out : reconstructed image in pixel space (B, C, H, W)
    """
    B, C, H, W = src_img.shape

    # 1. Fourier Transform & Shift
    F_src = torch.fft.fft2(src_img)
    F_shift = torch.fft.fftshift(F_src)

    # 2. Separate Amplitude and Phase
    amp_src = torch.abs(F_shift)
    phase_src = torch.angle(F_shift)

    # 3. Log-Scale the Amplitude
    # FFT amplitudes are massive. Log scaling compresses them so the GAN's
    # activation functions (Tanh/ReLU) don't instantly saturate and explode.
    amp_log = torch.log(amp_src + 1e-8)

    # 4. Generator Translation (on the full spatial dimension)
    # We pass the full size so the ResNet Generator maintains its receptive
    # field and internal dimensionality.
    amp_translated_log = G(amp_log)

    # 5. Inverse Log-Scale
    amp_translated = torch.exp(amp_translated_log)

    # 6. Create a Low-Frequency Mask based on beta
    h_crop = max(1, int(H * beta))
    w_crop = max(1, int(W * beta))
    h_start = H // 2 - h_crop // 2
    w_start = W // 2 - w_crop // 2

    mask = torch.zeros_like(amp_src)
    mask[:, :, h_start:h_start+h_crop, w_start:w_start+w_crop] = 1.0

    # 7. Blend: Use translated amplitude for low freqs, original for high freqs
    amp_new = (amp_translated * mask) + (amp_src * (1.0 - mask))

    # 8. Recombine with Original Phase
    F_new = amp_new * torch.exp(1j * phase_src)

    # 9. Inverse FFT back to spatial domain
    F_ishift = torch.fft.ifftshift(F_new)
    img_out = torch.fft.ifft2(F_ishift).real

    return img_out


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

    G_A2D = Generator().to(device)
    G_D2A = Generator().to(device)
    D_A   = Discriminator().to(device)
    D_D   = Discriminator().to(device)

    # Always initialise all weights first with Gaussian(0, 0.02)
    init_weights(G_A2D)
    init_weights(G_D2A)
    init_weights(D_A)
    init_weights(D_D)

    # Then optionally overwrite the encoder's first layer with pretrained weights
    if config.USE_PRETRAINED:
        load_pretrained_encoder(G_A2D)
        load_pretrained_encoder(G_D2A)
    else:
        print("[Scratch] Training fully from random initialisation.")

    criterion_GAN      = nn.MSELoss()
    criterion_cycle    = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_A2D.parameters(), G_D2A.parameters()),
        lr=config.LR, betas=(config.BETA1, config.BETA2),
    )
    optimizer_D_A = torch.optim.Adam(
        D_A.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2)
    )
    optimizer_D_D = torch.optim.Adam(
        D_D.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2)
    )

    lr_lambda     = build_lr_lambda(config.NUM_EPOCHS)
    scheduler_G   = torch.optim.lr_scheduler.LambdaLR(optimizer_G,   lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda)
    scheduler_D_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D_D, lr_lambda)

    buffer_D = ReplayBuffer(config.BUFFER_SIZE)
    buffer_A = ReplayBuffer(config.BUFFER_SIZE)

    beta = config.BETA_FREQ_BOTTLENECK

    # ---------------------------------------------------------- resume logic
    start_epoch = 0

    if config.RESUME:
        if config.RESUME_EPOCH == -1:
            ckpt_path = find_latest_checkpoint(config.CHECKPOINT_DIR + '/spectral')
        else:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR + '/spectral', f'epoch_{config.RESUME_EPOCH:03d}.pth'
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
    print(f"  Spectral CycleGAN — Office-31 Amazon → webcam")
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
        for batch_idx, (real_A, real_D) in enumerate(
                zip(loader_A, loader_D)):

            real_A = real_A.to(device)
            real_D = real_D.to(device)

            # ====================== Train Generators ======================
            optimizer_G.zero_grad()

            # Identity losses — always a direct G() call, never spectral_translate().
            # An amazon image fed to G_D2A (webcam→amazon) should come back unchanged,
            # and vice-versa. Wrapping this in spectral_translate would inject the
            # frequency blend and destroy the identity signal.
            loss_id_A = criterion_identity(G_D2A(real_A), real_A)
            loss_id_D = criterion_identity(G_A2D(real_D), real_D)

            # Forward translations
            fake_D = spectral_translate(G_A2D, real_A, beta)
            fake_A = spectral_translate(G_D2A, real_D, beta)

            # Adversarial losses
            loss_adv_A2D = criterion_GAN(D_D(fake_D),
                                         torch.ones_like(D_D(fake_D)))
            loss_adv_D2A = criterion_GAN(D_A(fake_A),
                                         torch.ones_like(D_A(fake_A)))

            # Cycle consistency losses
            rec_A = spectral_translate(G_D2A, fake_D, beta)
            rec_D = spectral_translate(G_A2D, fake_A, beta)
            loss_cycle_A = criterion_cycle(rec_A, real_A)
            loss_cycle_D = criterion_cycle(rec_D, real_D)

            loss_G = (
                loss_adv_A2D + loss_adv_D2A
                + config.LAMBDA_CYCLE    * (loss_cycle_A + loss_cycle_D)
                + config.LAMBDA_IDENTITY * (loss_id_A    + loss_id_D)
            )
            loss_G.backward()
            optimizer_G.step()

            # ================== Train Discriminator D_D ===================
            optimizer_D_D.zero_grad()

            fake_D_buf = buffer_D.push_and_pop(fake_D.detach())
            pred_real  = D_D(real_D)
            pred_fake  = D_D(fake_D_buf)
            loss_D_D = (
                criterion_GAN(pred_real, torch.ones_like(pred_real))
                + criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            ) * 0.5
            loss_D_D.backward()
            optimizer_D_D.step()

            # ================== Train Discriminator D_A ===================
            optimizer_D_A.zero_grad()

            fake_A_buf = buffer_A.push_and_pop(fake_A.detach())
            pred_real  = D_A(real_A)
            pred_fake  = D_A(fake_A_buf)
            loss_D_A = (
                criterion_GAN(pred_real, torch.ones_like(pred_real))
                + criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            ) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            if batch_idx % 50 == 0:
                print(
                    f'Epoch [{epoch:3d}/{config.NUM_EPOCHS}] '
                    f'Batch [{batch_idx:4d}] '
                    f'loss_G={loss_G.item():.4f}  '
                    f'loss_D_A={loss_D_A.item():.4f}  '
                    f'loss_D_D={loss_D_D.item():.4f}'
                )

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_D.step()

        if epoch % config.SAVE_EVERY == 0:
            with torch.no_grad():
                save_images(epoch, real_A, fake_D, real_D, fake_A,
                            save_dir=config.OUTPUT_DIR + '/spectral')

        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            save_checkpoint(
                epoch, G_A2D, G_D2A, D_A, D_D,
                optimizer_G, optimizer_D_A, optimizer_D_D,
                checkpoint_dir=config.CHECKPOINT_DIR + '/spectral',
                use_pretrained=config.USE_PRETRAINED,
            )

    print('Training complete.')


if __name__ == '__main__':
    train()
