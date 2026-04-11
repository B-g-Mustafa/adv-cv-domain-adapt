"""
Quantitative evaluation script: load a checkpoint and generate translated images
for an entire directory, mirroring the original subfolder structure.

Usage Examples:
    # Translate entire Amazon dataset to Webcam style using Spatial GAN
    python evaluate.py --checkpoint checkpoints/spatial/epoch_199.pth \
                       --mode spatial \
                       --direction A2D \
                       --input_dir data/office31/amazon/images \
                       --output_dir outputs/translated_to_webcam

    # Translate entire Webcam dataset to Amazon style using Spectral GAN
    python evaluate.py --checkpoint checkpoints/spectral/epoch_199.pth \
                       --mode spectral \
                       --direction D2A \
                       --input_dir data/office31/webcam/images \
                       --output_dir outputs/translated_to_amazon
"""

import argparse
import os
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

import config
from models import Generator
from utils import denorm
from train_spectral_bottleneck import spectral_translate

def load_generators(checkpoint_path, device):
    """Loads the saved weights into the Generator models."""
    G_A2D = Generator().to(device)
    G_D2A = Generator().to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    G_A2D.load_state_dict(ckpt['G_P2S'])  # Using the keys from your original checkpoint save
    G_D2A.load_state_dict(ckpt['G_S2P'])
    
    G_A2D.eval()
    G_D2A.eval()
    return G_A2D, G_D2A

def process_directory(checkpoint_path, input_dir, output_dir, mode, direction):
    """
    Recursively finds all images in input_dir, translates them, 
    and saves them to output_dir while maintaining folder structure.
    """
    # 1. Setup Device (Respecting your config, which should include MPS for Apple Silicon)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. Load Models
    print(f"Loading checkpoint: {checkpoint_path}")
    G_A2D, G_D2A = load_generators(checkpoint_path, device)
    
    # Select the correct generator based on translation direction
    if direction == 'A2D':
        G = G_A2D
        print("Direction: Amazon → Webcam (A2D)")
    else:
        G = G_D2A
        print("Direction: Webcam → Amazon (D2A)")

    # 3. Setup Image Transforms
    # CycleGAN evaluation typically uses resizing to 256x256, ToTensor, and Normalize.
    # Adjust these values if your config.py defines different dimensions.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 4. Path Handling
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    if not in_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
    out_path.mkdir(parents=True, exist_ok=True)

    # 5. Process Images
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in in_path.rglob('*') if f.suffix.lower() in valid_extensions]
    total_images = len(image_files)
    
    print(f"Found {total_images} images to process. Starting translation in '{mode}' mode...")

    for i, img_file in enumerate(image_files, 1):
        # Calculate relative path to maintain folder structure (e.g., 'monitor/frame_001.jpg')
        rel_path = img_file.relative_to(in_path)
        out_file = out_path / rel_path
        
        # Ensure the subfolder exists in the output directory
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Load and transform image
        try:
            img = Image.open(img_file).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device) # Add batch dimension
        except Exception as e:
            print(f"Skipping {img_file.name} due to read error: {e}")
            continue

        # Perform Translation
        with torch.no_grad():
            if mode == 'spatial':
                out_t = G(img_t)
            else: # spectral mode
                out_t = spectral_translate(G, img_t, config.BETA_FREQ_BOTTLENECK)

        # Save Image
        # We use denorm() to shift [-1, 1] back to [0, 1] before saving
        save_image(denorm(out_t).squeeze(0), out_file)

        # Print progress every 50 images
        if i % 50 == 0 or i == total_images:
            print(f"[{i:5d}/{total_images:5d}] Saved: {rel_path}")

    print(f"\n✅ Finished processing all {total_images} images!")
    print(f"Outputs are saved in: {out_path.absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translate entire datasets while preserving folder structure.")
    
    parser.add_argument('--checkpoint', required=True,
                        help='Path to the .pth checkpoint file')
    
    parser.add_argument('--input_dir', required=True,
                        help='Root directory of the dataset to translate (e.g., data/amazon/images)')
                        
    parser.add_argument('--output_dir', required=True,
                        help='Where to save the translated dataset structure')
                        
    parser.add_argument('--mode', choices=['spatial', 'spectral'], default='spatial',
                        help='Whether to use standard pixel translation or pure frequency translation')
                        
    parser.add_argument('--direction', choices=['A2D', 'D2A'], required=True,
                        help='A2D translates Amazon -> Webcam. D2A translates Webcam -> Amazon.')

    args = parser.parse_args()

    process_directory(
        checkpoint_path=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        direction=args.direction
    )