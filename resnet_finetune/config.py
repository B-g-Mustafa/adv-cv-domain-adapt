import torch

DATA_DIR     = 'data/'
OUTPUT_DIR   = 'outputs/'

IMAGE_SIZE   = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 20
LR           = 0.001
WEIGHT_DECAY = 1e-4

TRAIN_SPLIT  = 0.85
RANDOM_SEED  = 42

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

USE_PRETRAINED = True

# How many layers to freeze from the bottom of ResNet-50:
# 0 = train everything
# 1 = freeze conv1 + bn1
# 2 = freeze conv1 + bn1 + layer1
# 3 = freeze conv1 + bn1 + layer1 + layer2
# 4 = freeze conv1 + bn1 + layer1 + layer2 + layer3
FREEZE_LAYERS = 2
