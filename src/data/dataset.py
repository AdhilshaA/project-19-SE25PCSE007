"""
Dataset classes for metasurface inverse design.
Handles loading images and corresponding absorption spectra.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets as dset


class MetasurfaceDataset(Dataset):
    """
    PyTorch Dataset for metasurface inverse design.
    Loads image-spectrum pairs.
    """

    def __init__(
        self,
        img_dir,
        spectra_csv,
        image_size=64,
        n_spectrum_points=800,
        normalize=True,
    ):
        """
        Args:
            img_dir: Directory containing metasurface images
            spectra_csv: Path to CSV file with absorption spectra
            image_size: Size to resize images to (assumed square)
            n_spectrum_points: Number of spectrum points to use
            normalize: Whether to normalize images to [-1, 1]
        """
        self.image_size = image_size
        self.n_spectrum_points = n_spectrum_points
        self.normalize = normalize

        # Load spectra
        self.spectra_df = pd.read_csv(spectra_csv, header=0, index_col=0)
        self.spectra_values = self.spectra_df.iloc[:, :n_spectrum_points].values
        self.spectra_tensor = torch.from_numpy(self.spectra_values).float()

        # Get image filenames (sorted for consistency)
        self.image_dir = img_dir
        self.image_files = sorted(
            [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        # Build transform pipeline
        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        if normalize:
            # Normalize to [-1, 1]
            transform_list.append(transforms.Normalize([0.5], [0.5]))

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        from PIL import Image

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load corresponding spectrum
        spectrum = self.spectra_tensor[idx]

        return {
            "image": image,
            "spectrum": spectrum,
            "idx": idx,
        }


def get_dataloader(
    img_dir,
    spectra_csv,
    batch_size=16,
    num_workers=0,
    image_size=64,
    n_spectrum_points=800,
    shuffle=True,
    normalize=True,
):
    """
    Create a DataLoader for metasurface data.

    Args:
        img_dir: Directory containing images
        spectra_csv: Path to spectra CSV
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        image_size: Image size (square)
        n_spectrum_points: Number of spectrum bins
        shuffle: Whether to shuffle data
        normalize: Whether to normalize images

    Returns:
        DataLoader instance
    """
    dataset = MetasurfaceDataset(
        img_dir=img_dir,
        spectra_csv=spectra_csv,
        image_size=image_size,
        n_spectrum_points=n_spectrum_points,
        normalize=normalize,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def _split_lengths(n_items, train_ratio, val_ratio, test_ratio):
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("Split ratios must be non-negative")
    total = ratios.sum()
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")
    ratios = ratios / total
    raw = n_items * ratios
    lengths = np.floor(raw).astype(int)
    remainder = int(n_items - lengths.sum())
    if remainder > 0:
        fractional = raw - lengths
        order = np.argsort(-fractional)
        for idx in range(remainder):
            lengths[order[idx % len(order)]] += 1
    return tuple(int(length) for length in lengths)


def get_split_dataloaders(
    img_dir,
    spectra_csv,
    batch_size=16,
    num_workers=0,
    image_size=64,
    n_spectrum_points=800,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    manual_seed=999,
    normalize=True,
):
    """
    Create deterministic train/validation/test dataloaders.
    """
    dataset = MetasurfaceDataset(
        img_dir=img_dir,
        spectra_csv=spectra_csv,
        image_size=image_size,
        n_spectrum_points=n_spectrum_points,
        normalize=normalize,
    )

    train_len, val_len, test_len = _split_lengths(
        len(dataset), train_ratio, val_ratio, test_ratio
    )
    generator = torch.Generator().manual_seed(manual_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    split_sizes = {
        "train_size": train_len,
        "val_size": val_len,
        "test_size": test_len,
    }

    return train_loader, val_loader, test_loader, split_sizes
