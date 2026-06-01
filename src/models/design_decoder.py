"""
Design decoder: maps spatial latent codes back to geometry images.
"""

import torch
import torch.nn as nn


class DesignDecoder(nn.Module):
    """Latent-to-image decoder for metasurface geometry reconstruction."""

    def __init__(
        self,
        latent_channels=1,
        latent_size=16,
        hidden_dim=64,
        out_channels=3,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size

        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.SiLU(),
        )

        self.upsample_blocks = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(
                hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def _ensure_spatial(self, latent):
        if latent.dim() == 2:
            expected_dim = self.latent_channels * self.latent_size * self.latent_size
            if latent.shape[1] != expected_dim:
                raise ValueError(
                    f"Expected flat latent dimension {expected_dim}, got {latent.shape[1]}"
                )
            latent = latent.view(
                latent.shape[0],
                self.latent_channels,
                self.latent_size,
                self.latent_size,
            )
        elif latent.dim() != 4:
            raise ValueError(
                "latent must have shape (batch, dim) or (batch, channels, height, width)"
            )
        return latent

    def forward(self, latent):
        latent = self._ensure_spatial(latent)
        x = self.input_proj(latent)
        x = self.upsample_blocks(x)
        x = self.output_proj(x)
        return x


class DesignAutoencoder(nn.Module):
    """Convenience wrapper for jointly using the design encoder and decoder."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent = self.encoder(x)
        latent = self.encoder.to_spatial(latent)
        return self.decoder(latent)
