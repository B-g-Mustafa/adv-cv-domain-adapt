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
    Returns (loader_photo, loader_sketch).
    Expected structure:
        <data_root>/photo/<class>/img.jpg
        <data_root>/sketch/<class>/img.jpg
    """
    photo_dir  = os.path.join(data_root, 'photo')
    sketch_dir = os.path.join(data_root, 'sketch')

    transform = _build_transform()

    ds_photo  = FlatImageDataset(photo_dir,  transform)
    ds_sketch = FlatImageDataset(sketch_dir, transform)

    loader_photo = DataLoader(
        ds_photo,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )
    loader_sketch = DataLoader(
        ds_sketch,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )
    return loader_photo, loader_sketch
