import torch

IMAGE_SIZE      = 256
BATCH_SIZE      = 1
NUM_EPOCHS      = 200          # 100 fixed lr + 100 linear decay
LR              = 0.0002
BETA1           = 0.5          # Adam beta1
BETA2           = 0.999        # Adam beta2
LAMBDA_CYCLE    = 10.0
LAMBDA_IDENTITY = 5.0
BUFFER_SIZE     = 50
SAVE_EVERY      = 10           # save sample images every N epochs
NUM_WORKERS     = 2
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

# Spectral CycleGAN only
BETA_FREQ       = 0.01         # fraction of spectrum to translate (tune this)

# Paths
DATA_ROOT       = 'data/pacs'
OUTPUT_DIR      = 'outputs'
CHECKPOINT_DIR  = 'checkpoints'
SAVE_CHECKPOINT_EVERY = 20
