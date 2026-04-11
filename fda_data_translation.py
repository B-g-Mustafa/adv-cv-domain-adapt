import os
import torch
import torch.fft
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random

# --- CONFIGURATION ---
SOURCE_DIR = 'data/office31/amazon'
TARGET_DIR = 'data/office31/webcam'
OUTPUT_DIR = 'data/office31/fda_prepared'
BETA = 0.01  # Style intensity (0.01 to 0.05 is the sweet spot)

# Check if we can use the GPU (important for your Apple Silicon!)
device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")

def apply_fda_torch(src_tensor, trg_tensor, L=BETA):
    """
    Does the FFT swap entirely in PyTorch.
    Expects tensors of shape [C, H, W]
    """
    # 1. Fourier Transform
    fft_src = torch.fft.fft2(src_tensor)
    fft_trg = torch.fft.fft2(trg_tensor)

    # 2. Shift the low frequencies to the center
    fft_src_shift = torch.fft.fftshift(fft_src)
    fft_trg_shift = torch.fft.fftshift(fft_trg)

    # 3. Create the "window" for the swap
    _, h, w = src_tensor.shape
    b = int(min(h, w) * L)
    c_h, c_w = h // 2, w // 2

    # 4. Swap the amplitude (style) from target to source
    # We keep the Phase of the source (structural info)
    fft_src_shift[:, c_h-b:c_h+b, c_w-b:c_w+b] = fft_trg_shift[:, c_h-b:c_h+b, c_w-b:c_w+b]

    # 5. Inverse Shift and Inverse FFT
    img_back_shift = torch.fft.ifftshift(fft_src_shift)
    img_back = torch.fft.ifft2(img_back_shift)

    # Take the real part and ensure values are 0-1
    return torch.real(img_back).clamp(0, 1)

def prepare_fda_dataset():
    # Setup folders
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Standard transforms to get images into Tensors
    loader = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Transform to save back to disk
    to_pil = transforms.ToPILImage()

    print(f"Running FDA on {device}...")

    for cls in classes:
        src_cls_path = os.path.join(SOURCE_DIR, cls)
        trg_cls_path = os.path.join(TARGET_DIR, cls)
        out_cls_path = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(out_cls_path, exist_ok=True)

        src_images = [f for f in os.listdir(src_cls_path) if f.lower().endswith(('.png', '.jpg'))]
        trg_images = [f for f in os.listdir(trg_cls_path) if f.lower().endswith(('.png', '.jpg'))]

        for img_name in tqdm(src_images, desc=f"Styling {cls}"):
            # Load images as Tensors
            src_img = loader(Image.open(os.path.join(src_cls_path, img_name))).to(device)

            # Pick a random target style
            trg_img = loader(Image.open(os.path.join(trg_cls_path, random.choice(trg_images)))).to(device)

            # Apply FDA swap
            with torch.no_grad(): # We aren't training here, just processing
                result_tensor = apply_fda_torch(src_img, trg_img)

            # Save the result
            to_pil(result_tensor.cpu()).save(os.path.join(out_cls_path, img_name))

if __name__ == "__main__":
    prepare_fda_dataset()