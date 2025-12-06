import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LiverMRIDataset(Dataset):
    def __init__(self, root="data/processed", split="train", img_size=256):
        self.root = root
        self.split = split
        self.img_size = img_size

        self.img_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")

        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(".png")
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No PNG files found in {self.img_dir}. Did preprocessing run correctly?")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # Read in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Failed to read {img_path} or {mask_path}")

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Add channel dimension (C,H,W)
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image.float(), mask.float()
