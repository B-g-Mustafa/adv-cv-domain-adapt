import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


def _build_transform():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class FlatImageDataset(Dataset):
    """
    Loads all images under `root_dir` recursively, ignoring sub-folder labels.
    Supports .jpg, .jpeg, .png, .bmp, .webp extensions.
    """
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or _build_transform()
        self.paths = sorted([
            p for p in self.root_dir.rglob('*')
            if p.suffix.lower() in self.EXTENSIONS
        ])
        if len(self.paths) == 0:
            raise RuntimeError(f'No images found under {root_dir}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


def get_loaders(data_root=config.DATA_ROOT):
    """
    Returns (loader_A, loader_D) for Office-31 Amazon and webcam domains.
    Expected structure:
        <data_root>/amazon/<class>/img.jpg
        <data_root>/webcam/<class>/img.jpg
    """
    amazon_dir = os.path.join(data_root, 'amazon')
    webcam_dir   = os.path.join(data_root, 'webcam')

    transform = _build_transform()

    ds_amazon = FlatImageDataset(amazon_dir, transform)
    ds_webcam   = FlatImageDataset(webcam_dir,   transform)

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
