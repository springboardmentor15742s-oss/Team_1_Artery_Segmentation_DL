from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def paired_files(root):
    root = Path(root)
    image_dir, mask_dir = root / "images", root / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected {image_dir} and {mask_dir}. Read README.md for dataset layout."
        )
    images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in EXTS}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in EXTS}
    keys = sorted(set(images) & set(masks))
    if not keys:
        raise ValueError("No matching image/mask filename stems found.")
    missing_masks = sorted(set(images) - set(masks))
    if missing_masks:
        print(f"Warning: {len(missing_masks)} images have no matching masks and will be skipped.")
    return [(images[k], masks[k]) for k in keys]


def split_pairs(pairs, seed=42):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    if n < 3:
        raise ValueError("Need at least 3 paired images for train/validation/test splits.")
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    return pairs[:n_train], pairs[n_train:n_train+n_val], pairs[n_train+n_val:]


class VesselDataset(Dataset):
    def __init__(self, pairs, size=256, augment=False):
        self.pairs = pairs
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        x = np.asarray(image, dtype=np.float32) / 255.0
        y = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                x, y = np.fliplr(x).copy(), np.fliplr(y).copy()
            if random.random() < 0.5:
                x, y = np.flipud(x).copy(), np.flipud(y).copy()
        x = torch.from_numpy(x[None, ...])
        y = torch.from_numpy(y[None, ...])
        return x, y, image_path.stem


def get_device():
    # Explicit CPU-only setup for the requested hardware.
    return torch.device("cpu")
