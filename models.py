import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1,
                      padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1,
                      padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(channel),
        )

    def forward(self, x):
        return x + self.model(x)


class Generator(nn.Module):
    """
    ResNet-9 generator for 256×256 input.
    Architecture: c7s1-64, d128, d256, R256×9, u128, u64, c7s1-3
    Tensor flow:
        (B, 3, 256, 256) → (B, 64, 256, 256) → (B, 128, 128, 128)
        → (B, 256, 64, 64) → 9×R256 → (B, 128, 128, 128)
        → (B, 64, 256, 256) → (B, 3, 256, 256)
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Encoder (downsampling)
            nn.Conv2d(3, 64, 7, stride=1, padding=3,
                      padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1,
                      padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1,
                      padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # Residual blocks
            *[ResBlock(256) for _ in range(9)],

            # Decoder (upsampling) — no reflect padding on ConvTranspose2d
            nn.ConvTranspose2d(256, 128, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer — bias=True (default), no InstanceNorm, Tanh
            nn.Conv2d(64, 3, 7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    PatchGAN 70×70 discriminator.
    Output shape: (B, 1, 30, 30) — each cell scores one 70×70 patch.
    First layer has no InstanceNorm (intentional).
    Output layer has no activation (raw logits for MSELoss / LSGAN).
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1 — no InstanceNorm
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4 — stride 1
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 5 — output, no norm, no activation
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)   # shape: (B, 1, 30, 30)


def init_weights(net):
    """Initialise Conv/ConvTranspose with N(0, 0.02); InstanceNorm weight=1, bias=0."""
    def init_func(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
