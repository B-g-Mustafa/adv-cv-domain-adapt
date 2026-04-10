import torch.nn as nn
from torchvision import models

import config


# Load ResNet-50, replace the classification head, and freeze early layers.
def get_model(num_classes):
    if config.USE_PRETRAINED:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # ResNet-50 sections from shallowest to deepest:
    layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']

    for i in range(config.FREEZE_LAYERS):
        if i < len(layers_to_freeze):
            section = getattr(model, layers_to_freeze[i])
            for param in section.parameters():
                param.requires_grad = False

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f'Model ready.')
    print(f'  Total params    : {total:,}')
    print(f'  Trainable params: {trainable:,}')
    print(f'  Frozen params   : {frozen:,}')

    return model
