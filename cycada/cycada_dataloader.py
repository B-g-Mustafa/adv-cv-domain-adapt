import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# 1. Your exact logic to guarantee the label integers match your ResNet
def collect_images(data_dir):
    data_dir = Path(data_dir)
    class_names = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    if not class_names:
        raise RuntimeError(f'No class subfolders found in {data_dir}')

    all_paths = []
    all_labels = []

    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        paths = sorted([
            p for p in class_dir.rglob('*')
            if p.suffix.lower() in EXTENSIONS
        ])
        all_paths.extend(paths)
        all_labels.extend([label] * len(paths))

    return all_paths, all_labels, class_names


# 2. GAN-Specific Transforms
# Note: CycleGAN requires mean=0.5, std=0.5 to work with its Tanh output layer.
def build_gan_transform():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# 3. Your SimpleDataset class
class SimpleDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]


# 4. The CyCADA dual-loader function
def get_loaders(data_root=config.DATA_ROOT):
    """
    Returns (loader_A, loader_D) for CyCADA.
    Applies your mapping logic to both domains.
    """
    amazon_dir = os.path.join(data_root, 'amazon')
    webcam_dir = os.path.join(data_root, 'webcam')

    # Collect paths and labels using your established logic
    amazon_paths, amazon_labels, amazon_classes = collect_images(amazon_dir)
    webcam_paths, webcam_labels, webcam_classes = collect_images(webcam_dir)

    print(f"[Dataset] Amazon classes mapped: {len(amazon_classes)}")
    print(f"[Dataset] Webcam classes mapped: {len(webcam_classes)}")

    gan_transform = build_gan_transform()

    # We do NOT split train/test here, as UDA typically uses the whole
    # source and target datasets during the GAN translation phase.
    ds_amazon = SimpleDataset(amazon_paths, amazon_labels, gan_transform)
    ds_webcam = SimpleDataset(webcam_paths, webcam_labels, gan_transform)

    loader_A = DataLoader(
        ds_amazon,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )

    loader_D = DataLoader(
        ds_webcam,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )

    return loader_A, loader_D