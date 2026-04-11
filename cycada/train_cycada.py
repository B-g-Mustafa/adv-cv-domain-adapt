
import itertools
import os

import torch
import torch.nn as nn
from torchvision import transforms
from models import Generator, Discriminator, init_weights, load_pretrained_encoder
from resnet_finetune.predict import get_model

import config
try:
    from .cycada_dataloader import get_loaders
except ImportError:
    from cycada_dataloader import get_loaders
from utils import ReplayBuffer, save_images, save_checkpoint, find_latest_checkpoint, load_checkpoint


def build_lr_lambda(num_epochs=config.NUM_EPOCHS):
    """Keep lr constant for first half, then linearly decay to 0."""
    decay_start = num_epochs // 2

    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / float(decay_start))

    return lr_lambda


def load_frozen_classifier(device):
    """
    Loads the pre-trained ResNet, freezes its weights,
    and sets it to eval mode so it can grade the GAN.
    """
    save_path = config.RESNET_WEIGHTS
    if not os.path.exists(save_path):
        raise FileNotFoundError(f'Teacher model not found at {save_path}. Train ResNet first!')

    # Load checkpoint just like in predict.py
    checkpoint = torch.load(save_path, map_location=device)
    num_classes = checkpoint['num_classes']

    # Initialize model and load weights
    classifier = get_model(num_classes).to(device)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # FREEZE THE MODEL (Crucial step for CyCADA)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    print(f"[CyCADA] Loaded Frozen Teacher Model ({num_classes} classes)")
    return classifier


def get_gan_to_classifier_transform():
    """
    Converts the GAN's output format to the ResNet's input format.
    GAN output: normalized with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    ResNet input: normalized with ImageNet stats.
    """
    return transforms.Compose([
        # 1. Un-normalize GAN output back to standard 0-1 pixel values
        transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),

        # 2. Re-normalize to ImageNet stats (matching your predict.py)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train():
    device = torch.device(config.DEVICE)

    # ------------------------------------------------------------------ data
    loader_A, loader_D = get_loaders()
    print(f'Amazon images: {len(loader_A.dataset)}')
    print(f'webcam images: {len(loader_D.dataset)}')

    # ---------------------------------------------------------------- models
    G_A2D = Generator().to(device)  # Amazon → webcam
    G_D2A = Generator().to(device)  # webcam → Amazon
    D_A = Discriminator().to(device)
    D_D = Discriminator().to(device)

    # --- NEW: CyCADA Task Classifier (The "Teacher") ---
    source_classifier = load_frozen_classifier(device)
    gan_to_classifier = get_gan_to_classifier_transform()
    print("[CyCADA] Pre-trained Amazon Classifier loaded and frozen.")

    # Always initialise all GAN weights first
    init_weights(G_A2D)
    init_weights(G_D2A)
    init_weights(D_A)
    init_weights(D_D)

    if config.USE_PRETRAINED:
        load_pretrained_encoder(G_A2D)
        load_pretrained_encoder(G_D2A)
    else:
        print("[Scratch] Training fully from random initialisation.")

    # -------------------------------------------------------------- losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # --- NEW: Task Loss for Semantic Consistency ---
    criterion_task = nn.CrossEntropyLoss()

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

    lr_lambda = build_lr_lambda(config.NUM_EPOCHS)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda)
    scheduler_D_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D_D, lr_lambda)

    # --------------------------------------------------------- replay buffers
    buffer_D = ReplayBuffer(config.BUFFER_SIZE)
    buffer_A = ReplayBuffer(config.BUFFER_SIZE)

    # ---------------------------------------------------------- resume logic
    start_epoch = 0
    # ... (Keep your existing resume logic here, omitted for brevity) ...

    # ------------------------------------------ startup banner
    print("=" * 50)
    print(f"  CyCADA — Office-31 Amazon → webcam")
    print(f"  Semantic Task Loss : ENABLED")
    print("=" * 50)

    # ---------------------------------------------------------- training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):

        # --- NEW: Dataloader Unpacking ---
        # Assuming your dataset returns (image, label), we need the Amazon label now!
        for batch_idx, ((real_A, label_A), (real_D, _)) in enumerate(zip(loader_A, loader_D)):

            real_A = real_A.to(device)
            real_D = real_D.to(device)
            label_A = label_A.to(device)  # We need the label on the GPU/MPS too

            # ====================== Train Generators ======================
            optimizer_G.zero_grad()

            # Forward translations
            fake_D = G_A2D(real_A)
            fake_A = G_D2A(real_D)

            # Identity losses
            loss_id_A = criterion_identity(G_D2A(real_A), real_A)
            loss_id_D = criterion_identity(G_A2D(real_D), real_D)

            # Adversarial losses
            loss_adv_A2D = criterion_GAN(D_D(fake_D), torch.ones_like(D_D(fake_D)))
            loss_adv_D2A = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))

            # Cycle consistency losses
            rec_A = G_D2A(fake_D)
            rec_D = G_A2D(fake_A)
            loss_cycle_A = criterion_cycle(rec_A, real_A)
            loss_cycle_D = criterion_cycle(rec_D, real_D)

            # --- NEW: CyCADA Semantic Consistency (Task Loss) ---
            # Ask the frozen classifier: "What object is in this fake webcam image?"
            classifier_img = gan_to_classifier(fake_D)
            pred_fake_D = source_classifier(classifier_img)

            # Penalize the Generator if the classifier gets the wrong label
            loss_task = criterion_task(pred_fake_D, label_A)

            # Total Generator Loss
            # Note: You may need to add LAMBDA_TASK (e.g., 1.0) to your config.py
            loss_G = (
                    loss_adv_A2D + loss_adv_D2A
                    + config.LAMBDA_CYCLE * (loss_cycle_A + loss_cycle_D)
                    + config.LAMBDA_IDENTITY * (loss_id_A + loss_id_D)
                    + config.LAMBDA_TASK * loss_task  # The CyCADA magic
            )
            loss_G.backward()
            optimizer_G.step()

            # ================== Train Discriminators (Same as before) ===================
            # (Discriminator D_D and D_A logic remains exactly the same)
            optimizer_D_D.zero_grad()
            fake_D_buf = buffer_D.push_and_pop(fake_D.detach())
            loss_D_D = (criterion_GAN(D_D(real_D), torch.ones_like(D_D(real_D))) +
                        criterion_GAN(D_D(fake_D_buf), torch.zeros_like(D_D(fake_D_buf)))) * 0.5
            loss_D_D.backward()
            optimizer_D_D.step()

            optimizer_D_A.zero_grad()
            fake_A_buf = buffer_A.push_and_pop(fake_A.detach())
            loss_D_A = (criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A))) +
                        criterion_GAN(D_A(fake_A_buf), torch.zeros_like(D_A(fake_A_buf)))) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # ----------------------- logging ------------------------------
            if batch_idx % 50 == 0:
                print(
                    f'Epoch [{epoch:3d}/{config.NUM_EPOCHS}] '
                    f'Batch [{batch_idx:4d}] '
                    f'loss_G={loss_G.item():.4f} '
                    f'(Task={loss_task.item():.4f}) '  # Log the task loss to monitor it
                    f'loss_D_A={loss_D_A.item():.4f} '
                    f'loss_D_D={loss_D_D.item():.4f}'
                )

        # Step LR schedulers and save logic remains unchanged...
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_D.step()

        # ====================== SAVE SAMPLE IMAGES ======================
        if epoch % config.SAVE_EVERY == 0:
            with torch.no_grad():
                save_images(epoch, real_A, fake_D, real_D, fake_A,
                            # UPDATE: Changed directory to separate from baseline CycleGAN
                            save_dir=config.OUTPUT_DIR + '/cycada')

        # ====================== SAVE CHECKPOINTS ======================
        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            # Note: We intentionally do NOT save `source_classifier` here.
            # Its weights are frozen, so the original pre-trained file is all you need.
            save_checkpoint(
                epoch, G_A2D, G_D2A, D_A, D_D,
                optimizer_G, optimizer_D_A, optimizer_D_D,
                # UPDATE: Changed directory to separate from baseline CycleGAN
                checkpoint_dir=config.CHECKPOINT_DIR + '/cycada',
                use_pretrained=config.USE_PRETRAINED,
            )

    print('CyCADA Training complete.')


if __name__ == '__main__':
    train()