"""
Forward predictor: maps geometry images to absorption spectra.
Used for training and for surrogate guidance in diffusion.
"""

import torch
import torch.nn as nn


class ForwardPredictor(nn.Module):
    """
    Simple forward predictor: image (geometry) -> spectrum.
    Uses a CNN encoder followed by FC decoder to output 1D spectrum.
    """

    def __init__(
        self,
        img_size=64,
        in_channels=3,
        hidden_dim=64,
        n_spectrum_points=800,
    ):
        """
        Args:
            img_size: Input image size (assumed square)
            in_channels: Number of input channels (3 for RGB)
            hidden_dim: Base hidden dimension
            n_spectrum_points: Number of spectrum bins to predict
        """
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_spectrum_points = n_spectrum_points

        # CNN encoder
        self.conv_encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(
                hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(
                hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True),
        )

        # After 4 strided convolutions: 64 -> 4
        self.flatten_dim = hidden_dim * 8 * 4 * 4

        # FC decoder to spectrum
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_spectrum_points),
            nn.Sigmoid(),  # Normalize output to [0, 1] to match spectrum range
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, img_size, img_size)
        Returns:
            spectrum: (batch_size, n_spectrum_points)
        """
        features = self.conv_encoder(x)
        features_flat = features.view(features.shape[0], -1)
        spectrum = self.fc_decoder(features_flat)
        return spectrum
