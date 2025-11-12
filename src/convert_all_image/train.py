import argparse
from pathlib import Path
import os
import sys

# Allow running the script directly without PYTHONPATH configured
if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import yaml

from data.datasets import SegmentationDataset
from models.unet import UNet


def build_transforms(img_size=(256, 256)):
    image_transform = T.Compose([
        T.Resize(img_size, antialias=True),
        T.ToTensor(),
    ])
    # Use nearest for masks to preserve label indices
    mask_transform = T.Compose([
        T.Resize(img_size, interpolation=Image.NEAREST),
        T.PILToTensor(),  # -> [1, H, W], uint8
        T.Lambda(lambda x: x.squeeze(0).to(torch.int64)),
    ])
    return image_transform, mask_transform


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train UNet for segmentation")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to YAML config with image_directories/mask_directories")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--img-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="cuda",
                        help="Select device: auto, cpu, or cuda")
    parser.add_argument("--disable-cudnn", action="store_true",
                        help="Disable cuDNN (workaround for some GPU backend errors)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Limit training steps per epoch (0 = all)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    image_dirs = cfg.get("image_directories", [])
    mask_dirs = cfg.get("mask_directories", [])
    if not image_dirs or not mask_dirs:
        raise ValueError("Config must define non-empty image_directories and mask_directories lists")

    img_dir = image_dirs[0]
    mask_dir = mask_dirs[0]

    image_transform, mask_transform = build_transforms(tuple(args.img_size))

    dataset = SegmentationDataset(
        img_dir,
        mask_dir,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if args.max_steps and (step_idx + 1) >= args.max_steps:
                break

        avg = epoch_loss / max(min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader), 1)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg:.4f}")


if __name__ == "__main__":
    main()
