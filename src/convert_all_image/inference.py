import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Allow running the script directly without PYTHONPATH configured
if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from convert_all_image.models.unet import UNet


def predict(model: torch.nn.Module, image_path: str, device: torch.device, size=(256, 256)):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = T.Resize(size, antialias=True)(image)
    image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        return np.array(image), pred_mask


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--weights", type=str, required=False, help="Path to model weights (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--img-size", type=int, nargs=2, default=(256, 256))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)

    if args.weights:
        state = torch.load(args.weights, map_location=device)
        # Support both raw state_dict and checkpoint dicts
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)

    original, pred = predict(model, args.image, device, size=tuple(args.img_size))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred, cmap="tab10")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
