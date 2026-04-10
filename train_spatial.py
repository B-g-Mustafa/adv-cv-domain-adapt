"""
Spatial CycleGAN training script.
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


def build_lr_lambda(num_epochs=config.NUM_EPOCHS):
    """Keep lr constant for first half, then linearly decay to 0."""
    decay_start = num_epochs // 2

    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / float(decay_start))

    return lr_lambda


def train():
    device = torch.device(config.DEVICE)

    # ------------------------------------------------------------------ data
    loader_A, loader_D = get_loaders()
    print(f'Amazon images: {len(loader_A.dataset)}')
    print(f'webcam images  : {len(loader_D.dataset)}')

    # ---------------------------------------------------------------- models
    G_A2D = Generator().to(device)   # Amazon → webcam
    G_D2A = Generator().to(device)   # webcam → Amazon
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

    # -------------------------------------------------------------- losses
    criterion_GAN      = nn.MSELoss()
    criterion_cycle    = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # ------------------------------------------------------------ optimisers
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

    # --------------------------------------------------------- replay buffers
    buffer_D = ReplayBuffer(config.BUFFER_SIZE)
    buffer_A = ReplayBuffer(config.BUFFER_SIZE)

    # ---------------------------------------------------------- resume logic
    start_epoch = 0

    if config.RESUME:
        if config.RESUME_EPOCH == -1:
            ckpt_path = find_latest_checkpoint(config.CHECKPOINT_DIR + '/spatial')
        else:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR + '/spatial', f'epoch_{config.RESUME_EPOCH:03d}.pth'
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
    print(f"  CycleGAN — Office-31 Amazon → webcam")
    print(f"  Pretrained encoder : {config.USE_PRETRAINED}")
    print(f"  Device             : {config.DEVICE}")
    print(f"  Epochs             : {config.NUM_EPOCHS}")
    print(f"  Resume             : {config.RESUME}")
    if config.RESUME:
        mode = 'latest' if config.RESUME_EPOCH == -1 else f'epoch {config.RESUME_EPOCH}'
        print(f"  Resume mode        : {mode}")
    print(f"  Starting epoch     : {start_epoch}")
    print("=" * 50)

    # ---------------------------------------------------------- training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        for batch_idx, (real_A, real_D) in enumerate(
                zip(loader_A, loader_D)):

            real_A = real_A.to(device)
            real_D = real_D.to(device)

            # ====================== Train Generators ======================
            optimizer_G.zero_grad()

            # Identity losses — G_D2A given an amazon image should return amazon
            loss_id_A = criterion_identity(G_D2A(real_A), real_A)
            loss_id_D = criterion_identity(G_A2D(real_D), real_D)

            # Forward translations
            fake_D = G_A2D(real_A)
            fake_A = G_D2A(real_D)

            # Adversarial losses (fool the discriminators)
            loss_adv_A2D = criterion_GAN(D_D(fake_D),
                                         torch.ones_like(D_D(fake_D)))
            loss_adv_D2A = criterion_GAN(D_A(fake_A),
                                         torch.ones_like(D_A(fake_A)))

            # Cycle consistency losses
            rec_A = G_D2A(fake_D)
            rec_D = G_A2D(fake_A)
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

            # ----------------------- logging ------------------------------
            if batch_idx % 50 == 0:
                print(
                    f'Epoch [{epoch:3d}/{config.NUM_EPOCHS}] '
                    f'Batch [{batch_idx:4d}] '
                    f'loss_G={loss_G.item():.4f}  '
                    f'loss_D_A={loss_D_A.item():.4f}  '
                    f'loss_D_D={loss_D_D.item():.4f}'
                )

        # Step LR schedulers after each epoch
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_D.step()

        # Save sample images
        if epoch % config.SAVE_EVERY == 0:
            with torch.no_grad():
                save_images(epoch, real_A, fake_D, real_D, fake_A,
                            save_dir=config.OUTPUT_DIR + '/spatial')

        # Save model checkpoints
        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            save_checkpoint(
                epoch, G_A2D, G_D2A, D_A, D_D,
                optimizer_G, optimizer_D_A, optimizer_D_D,
                checkpoint_dir=config.CHECKPOINT_DIR + '/spatial',
                use_pretrained=config.USE_PRETRAINED,
            )

    print('Training complete.')


if __name__ == '__main__':
    train()
