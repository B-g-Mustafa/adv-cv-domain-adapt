import os
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from resnet_finetune import config

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# Scan DATA_DIR and collect all image paths with integer class labels.
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

    print(f'Found {len(class_names)} classes: {", ".join(class_names)}')

    total = 0
    per_class = {}
    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        paths = sorted([
            p for p in class_dir.rglob('*')
            if p.suffix.lower() in EXTENSIONS
        ])
        per_class[class_name] = len(paths)
        total += len(paths)
        all_paths.extend(paths)
        all_labels.extend([label] * len(paths))

    print(f'Total images: {total}')
    for class_name, count in per_class.items():
        print(f'  {class_name:<12}: {count} images')

    return all_paths, all_labels, class_names


# Build train and test transforms (ImageNet stats because we use pretrained weights).
def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, test_transform


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


# Build and return train/test DataLoaders plus class metadata.
def get_dataloaders(data_dir=config.DATA_DIR):
    all_paths, all_labels, class_names = collect_images(data_dir)
    num_classes = len(class_names)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=1 - config.TRAIN_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=all_labels,
    )

    print(f'Train: {len(train_paths)} images')
    print(f'Test:  {len(test_paths)} images')

    train_transform, test_transform = build_transforms()

    train_loader = DataLoader(
        SimpleDataset(train_paths, train_labels, train_transform),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        SimpleDataset(test_paths, test_labels, test_transform),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, test_loader, class_names, num_classes
