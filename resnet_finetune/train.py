import os

import torch
import torch.nn as nn

import config
from dataset import get_dataloaders
from model import get_model


# Run one full pass over the training set and return average loss and accuracy.
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# Evaluate the model on the given loader and return accuracy.
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)

    return correct / total


def train():
    device = torch.device(config.DEVICE)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_loader, test_loader, class_names, num_classes = get_dataloaders()
    model = get_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_accuracy = 0.0
    best_epoch = 0
    save_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_accuracy = evaluate(model, test_loader, device)
        scheduler.step(test_accuracy)

        current_lr = optimizer.param_groups[0]['lr']
        marker = ''

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            marker = ' * best'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_accuracy,
                'class_names': class_names,
                'num_classes': num_classes,
            }, save_path)

        print(
            f'Epoch {epoch:2d}/{config.NUM_EPOCHS} | '
            f'Train loss: {train_loss:.4f} acc: {train_accuracy*100:.1f}% | '
            f'Test acc: {test_accuracy*100:.1f}% | '
            f'LR: {current_lr:.6f}'
            f'{marker}'
        )

    print()
    print('Training complete.')
    print(f'Best test accuracy: {best_accuracy*100:.1f}% at epoch {best_epoch}')
    print(f'Model saved to: {save_path}')


if __name__ == '__main__':
    train()


# HOW TO USE
# ----------
# 1. Put your data in data/ with one subfolder per class
# 2. Edit config.py if needed (DATA_DIR, NUM_EPOCHS etc.)
# 3. Run training:
#      python train.py
# 4. Evaluate on test set:
#      python evaluate.py
# 5. Predict a single image:
#      python predict.py path/to/your/image.jpg
