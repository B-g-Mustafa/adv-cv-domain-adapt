import os
import sys

import torch
from PIL import Image
from torchvision import transforms

import config
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

    return model, class_names


# Preprocess a single image file into a model-ready tensor.
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)


# Run inference on one image and print the predicted class with confidence scores.
def predict(image_path):
    if not os.path.exists(image_path):
        print(f'Error: file not found: {image_path}')
        sys.exit(1)

    device = torch.device(config.DEVICE)
    model, class_names = load_checkpoint(device)

    tensor = preprocess(image_path).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze()

    sorted_indices = probabilities.argsort(descending=True)

    print(f'Image   : {image_path}')
    print('\u2500' * 34)

    bar_max = 20
    for index in sorted_indices:
        class_name = class_names[index]
        percentage = probabilities[index].item() * 100
        bar_length = int(percentage / 100 * bar_max)
        bar = '\u2588' * bar_length
        print(f'{class_name:<10} {percentage:5.1f}%  {bar}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python predict.py path/to/image.jpg')
        sys.exit(1)

    predict(sys.argv[1])
