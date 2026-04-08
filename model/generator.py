import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(channel)
        )
    def forward(self, x):
        return x + self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # DownSampling
            nn.Conv2d(3, 64, 7, stride=1, padding=3, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(),

            #ResBlocks
            *[ResBlock(256) for _ in range(9)],

            #UpSampling
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 3, 7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
    def forward(self,x):
        return self.model(x)

#for Conv2d bias=False coz IN2D layer's bias makes the Conv2d redundant.