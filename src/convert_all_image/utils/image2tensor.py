"""Image to tensor conversion utility."""

import argparse
import torch
import numpy as np
from PIL import Image


def image_to_tensor(image_path: str) -> torch.Tensor:
    """Convert an image to a PyTorch tensor (no normalization)."""
    image = Image.open(image_path)
    array = np.array(image)
    tensor = torch.from_numpy(array)
    return tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    t = image_to_tensor(args.image)
    print(f"Tensor shape: {tuple(t.shape)}; dtype: {t.dtype}")
