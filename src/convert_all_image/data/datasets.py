import os
from typing import Optional, Callable, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    """Paired image/mask dataset for semantic segmentation.

    Expects two folders with equal-length, sorted lists of files.
    Transforms for image and mask are applied separately; prefer
    deterministic transforms (e.g., Resize) to keep pairs aligned.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        valid_image_exts: Optional[List[str]] = None,
        valid_mask_exts: Optional[List[str]] = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        if valid_image_exts is None:
            valid_image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        if valid_mask_exts is None:
            valid_mask_exts = [".png", ".bmp", ".tif", ".tiff"]

        def list_files(root: str, exts: List[str]) -> List[str]:
            items = [
                f
                for f in os.listdir(root)
                if not f.startswith(".") and os.path.splitext(f)[1].lower() in exts
            ]
            return sorted(items)

        self.images = list_files(self.image_dir, valid_image_exts)
        self.masks = list_files(self.mask_dir, valid_mask_exts)

        assert len(self.images) == len(
            self.masks
        ), f"Mismatched counts: {len(self.images)} images vs {len(self.masks)} masks"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            image = T.ToTensor()(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask
