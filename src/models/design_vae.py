"""
Convolutional VAE for design latent space (optional ablation).
"""

import torch
import torch.nn as nn


class ConvVAEEncoder(nn.Module):
    """Encoder that outputs spatial mu and logvar for a conv-VAE.

    Produces flat mu/logvar of size latent_channels*latent_size*latent_size
    to be reshaped by `to_spatial`.
    """

    def __init__(
        self,
        img_size=64,
        in_channels=3,
        hidden_dim=64,
        latent_channels=1,
        latent_size=16,
        logvar_min=-6.0,
        logvar_max=2.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.spatial_latent_dim = latent_channels * latent_size * latent_size
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        # Conv feature extractor (same footprint as DesignEncoder)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True),
        )

        self.flatten_dim = hidden_dim * 8 * 4 * 4

        # Project to mu and logvar of the spatial latent dimension
        self.fc_mu = nn.Linear(self.flatten_dim, self.spatial_latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.spatial_latent_dim)

    def forward(self, x):
        features = self.conv_encoder(x)
        features_flat = features.view(features.shape[0], -1)
        mu = self.fc_mu(features_flat)
        # Clamp log-variance for numerical stability.
        logvar = self.fc_logvar(features_flat).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def to_spatial(self, flat):
        if flat.dim() != 2:
            raise ValueError("flat must have shape (batch, dim)")
        if flat.shape[1] != self.spatial_latent_dim:
            raise ValueError(
                f"latent dim must match {self.spatial_latent_dim}, got {flat.shape[1]}"
            )
        return flat.view(
            flat.shape[0], self.latent_channels, self.latent_size, self.latent_size
        )


class ConvVAE(nn.Module):
    """Wrapper VAE that pairs the encoder with an existing decoder.

    Decoder should accept spatial latent maps `(B, C, H, W)`.
    """

    def __init__(self, encoder: ConvVAEEncoder, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, sample=True):
        mu, logvar = self.encoder(x)
        if sample:
            z = self.encoder.reparameterize(mu, logvar)
        else:
            z = mu
        z_spatial = self.encoder.to_spatial(z)
        recon = self.decoder(z_spatial)
        return recon, mu, logvar, z

    @staticmethod
    def kl_loss(mu, logvar, free_bits=0.0, normalize_by_dim=True):
        """KL divergence term with optional free-bits and dimensional normalization.

        free_bits is applied per latent dimension in nats. For stable VAE training,
        normalize_by_dim=True keeps KL scale comparable when latent size changes.
        """
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if free_bits > 0:
            kld = torch.clamp(kld, min=float(free_bits))
        if normalize_by_dim:
            kld_per_sample = torch.mean(kld, dim=1)
        else:
            kld_per_sample = torch.sum(kld, dim=1)
        return torch.mean(kld_per_sample)
