import os

import torch

import config
from dataset import get_dataloaders
from model import get_model


# Load the saved checkpoint and restore model weights and class names.
def load_checkpoint(device):
    save_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    if not os.path.exists(save_path):
        raise FileNotFoundError(f'No checkpoint found at {save_path}. Run train.py first.')

    checkpoint = torch.load(save_path, map_location=device)
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']

    model = get_model(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]} '
          f'(best acc: {checkpoint["best_acc"]*100:.1f}%)')

    return model, class_names, num_classes


# Evaluate the model on the test set and print per-class and overall accuracy.
def evaluate():
    device = torch.device(config.DEVICE)
    model, class_names, num_classes = load_checkpoint(device)

    _, test_loader, _, _ = get_dataloaders()

    # Collect predictions for each class
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes
    confusion      = [[0] * num_classes for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            for true_label, predicted_label in zip(labels, predictions):
                true_label      = true_label.item()
                predicted_label = predicted_label.item()
                class_total[true_label]   += 1
                confusion[true_label][predicted_label] += 1
                if true_label == predicted_label:
                    class_correct[true_label] += 1

    # Per-class accuracy table
    print()
    print(f'{"Class":<15} {"Total":>6}  {"Correct":>7}  {"Accuracy":>9}')
    print('-' * 42)

    overall_correct = sum(class_correct)
    overall_total   = sum(class_total)

    for i, class_name in enumerate(class_names):
        accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        print(f'{class_name:<15} {class_total[i]:>6}  {class_correct[i]:>7}  {accuracy*100:>8.1f}%')

    print('\u2500' * 42)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    print(f'{"Overall":<15} {overall_total:>6}  {overall_correct:>7}  {overall_accuracy*100:>8.1f}%')

    # Confusion matrix
    print()
    print('Confusion matrix (rows = true, cols = predicted):')
    col_width = max(len(name) for name in class_names) + 2

    header = ' ' * col_width + ''.join(f'{name:>{col_width}}' for name in class_names)
    print(header)
    print('-' * len(header))

    for i, class_name in enumerate(class_names):
        row = f'{class_name:<{col_width}}'
        row += ''.join(f'{confusion[i][j]:>{col_width}}' for j in range(num_classes))
        print(row)


if __name__ == '__main__':
    evaluate()
