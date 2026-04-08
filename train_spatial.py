"""
Spatial CycleGAN training script.
Photo ↔ Sketch domain adaptation on the PACS dataset.
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
    loader_photo, loader_sketch = get_loaders()
    print(f'Photo images : {len(loader_photo.dataset)}')
    print(f'Sketch images: {len(loader_sketch.dataset)}')

    # ---------------------------------------------------------------- models
    G_P2S = Generator().to(device)   # Photo → Sketch
    G_S2P = Generator().to(device)   # Sketch → Photo
    D_P   = Discriminator().to(device)
    D_S   = Discriminator().to(device)

    # Always initialise all weights first with Gaussian(0, 0.02)
    init_weights(G_P2S)
    init_weights(G_S2P)
    init_weights(D_P)
    init_weights(D_S)

    # Then optionally overwrite the encoder's first layer with pretrained weights
    if config.USE_PRETRAINED:
        load_pretrained_encoder(G_P2S)
        load_pretrained_encoder(G_S2P)
    else:
        print("[Scratch] Training fully from random initialisation.")

    # -------------------------------------------------------------- losses
    criterion_GAN      = nn.MSELoss()
    criterion_cycle    = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # ------------------------------------------------------------ optimisers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_P2S.parameters(), G_S2P.parameters()),
        lr=config.LR, betas=(config.BETA1, config.BETA2),
    )
    optimizer_D_P = torch.optim.Adam(
        D_P.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2)
    )
    optimizer_D_S = torch.optim.Adam(
        D_S.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2)
    )

    lr_lambda     = build_lr_lambda(config.NUM_EPOCHS)
    scheduler_G   = torch.optim.lr_scheduler.LambdaLR(optimizer_G,   lr_lambda)
    scheduler_D_P = torch.optim.lr_scheduler.LambdaLR(optimizer_D_P, lr_lambda)
    scheduler_D_S = torch.optim.lr_scheduler.LambdaLR(optimizer_D_S, lr_lambda)

    # --------------------------------------------------------- replay buffers
    buffer_S = ReplayBuffer(config.BUFFER_SIZE)
    buffer_P = ReplayBuffer(config.BUFFER_SIZE)

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
                G_P2S, G_S2P, D_P, D_S,
                optimizer_G, optimizer_D_P, optimizer_D_S,
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
            scheduler_D_P.step()
            scheduler_D_S.step()
        print(f"[Resume] LR schedulers fast-forwarded to epoch {start_epoch}.")

    # ------------------------------------------ startup banner (post-resume)
    print("=" * 50)
    print(f"  CycleGAN — PACS Photo → Sketch")
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
        for batch_idx, (real_P, real_S) in enumerate(
                zip(loader_photo, loader_sketch)):

            real_P = real_P.to(device)
            real_S = real_S.to(device)

            # ====================== Train Generators ======================
            optimizer_G.zero_grad()

            # Identity losses — G_S2P given a photo should return a photo
            loss_id_P = criterion_identity(G_S2P(real_P), real_P)
            loss_id_S = criterion_identity(G_P2S(real_S), real_S)

            # Forward translations
            fake_S = G_P2S(real_P)
            fake_P = G_S2P(real_S)

            # Adversarial losses (fool the discriminators)
            loss_adv_P2S = criterion_GAN(D_S(fake_S),
                                         torch.ones_like(D_S(fake_S)))
            loss_adv_S2P = criterion_GAN(D_P(fake_P),
                                         torch.ones_like(D_P(fake_P)))

            # Cycle consistency losses
            rec_P = G_S2P(fake_S)
            rec_S = G_P2S(fake_P)
            loss_cycle_P = criterion_cycle(rec_P, real_P)
            loss_cycle_S = criterion_cycle(rec_S, real_S)

            loss_G = (
                loss_adv_P2S + loss_adv_S2P
                + config.LAMBDA_CYCLE    * (loss_cycle_P + loss_cycle_S)
                + config.LAMBDA_IDENTITY * (loss_id_P    + loss_id_S)
            )
            loss_G.backward()
            optimizer_G.step()

            # ================== Train Discriminator D_S ===================
            optimizer_D_S.zero_grad()

            fake_S_buf = buffer_S.push_and_pop(fake_S.detach())
            pred_real  = D_S(real_S)
            pred_fake  = D_S(fake_S_buf)
            loss_D_S = (
                criterion_GAN(pred_real, torch.ones_like(pred_real))
                + criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            ) * 0.5
            loss_D_S.backward()
            optimizer_D_S.step()

            # ================== Train Discriminator D_P ===================
            optimizer_D_P.zero_grad()

            fake_P_buf = buffer_P.push_and_pop(fake_P.detach())
            pred_real  = D_P(real_P)
            pred_fake  = D_P(fake_P_buf)
            loss_D_P = (
                criterion_GAN(pred_real, torch.ones_like(pred_real))
                + criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            ) * 0.5
            loss_D_P.backward()
            optimizer_D_P.step()

            # ----------------------- logging ------------------------------
            if batch_idx % 50 == 0:
                print(
                    f'Epoch [{epoch:3d}/{config.NUM_EPOCHS}] '
                    f'Batch [{batch_idx:4d}] '
                    f'loss_G={loss_G.item():.4f}  '
                    f'loss_D_P={loss_D_P.item():.4f}  '
                    f'loss_D_S={loss_D_S.item():.4f}'
                )

        # Step LR schedulers after each epoch
        scheduler_G.step()
        scheduler_D_P.step()
        scheduler_D_S.step()

        # Save sample images
        if epoch % config.SAVE_EVERY == 0:
            with torch.no_grad():
                save_images(epoch, real_P, fake_S, real_S, fake_P,
                            save_dir=config.OUTPUT_DIR + '/spatial')

        # Save model checkpoints
        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            save_checkpoint(
                epoch, G_P2S, G_S2P, D_P, D_S,
                optimizer_G, optimizer_D_P, optimizer_D_S,
                checkpoint_dir=config.CHECKPOINT_DIR + '/spatial',
                use_pretrained=config.USE_PRETRAINED,
            )

    print('Training complete.')


if __name__ == '__main__':
    train()
