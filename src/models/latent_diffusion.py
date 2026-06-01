"""
Latent diffusion denoiser: U-Net-like architecture with cross-attention fusion.
Attends to spectrum tokens during denoising.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for time steps."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """
        Args:
            t: (batch_size,) time steps in [0, 1]
        Returns:
            embedding: (batch_size, embed_dim)
        """
        # Scale to match typical diffusion timestep scales
        t_scaled = t * 1000  # Scale to [0, 1000] range

        device = t.device
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t_scaled[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with optional cross-attention."""

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        use_attention=False,
        n_attention_heads=4,
        spectrum_token_dim=None,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            time_emb_dim: Time embedding dimension
            use_attention: Whether to use self-attention
            n_attention_heads: Number of attention heads
            spectrum_token_dim: Dimension of spectrum tokens (for cross-attention)
        """
        super().__init__()

        self.use_attention = use_attention

        self.norm1 = nn.GroupNorm(
            num_groups=min(8, in_channels), num_channels=in_channels
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            num_groups=min(8, out_channels), num_channels=out_channels
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

        # Skip connection projection
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = None

        # Self-attention (optional)
        if use_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=n_attention_heads,
                batch_first=True,
            )

        # Cross-attention to spectrum tokens (optional)
        if spectrum_token_dim is not None:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=n_attention_heads,
                kdim=spectrum_token_dim,
                vdim=spectrum_token_dim,
                batch_first=True,
            )
        else:
            self.cross_attention = None

        self.act = nn.SiLU()

    def forward(self, x, t_emb, spectrum_tokens=None):
        """
        Args:
            x: (batch_size, in_channels, h, w)
            t_emb: (batch_size, time_emb_dim)
            spectrum_tokens: (batch_size, n_tokens, spectrum_token_dim) or None
        Returns:
            out: (batch_size, out_channels, h, w)
        """
        residual = x
        h = self.conv1(self.act(self.norm1(x)))

        # Add time embedding
        t_proj = self.time_mlp(t_emb)
        h = h + t_proj[:, :, None, None]

        h = self.conv2(self.act(self.norm2(h)))

        # Self-attention (optional)
        if self.use_attention:
            batch_size, channels, height, width = h.shape
            h_flat = h.view(batch_size, channels, -1).transpose(1, 2)  # (B, HW, C)
            h_attn, _ = self.self_attention(h_flat, h_flat, h_flat)
            h = h + h_attn.transpose(1, 2).view(batch_size, channels, height, width)

        # Cross-attention to spectrum (optional)
        if self.cross_attention is not None and spectrum_tokens is not None:
            batch_size, channels, height, width = h.shape
            h_flat = h.view(batch_size, channels, -1).transpose(1, 2)  # (B, HW, C)
            h_cross, _ = self.cross_attention(h_flat, spectrum_tokens, spectrum_tokens)
            h = h + h_cross.transpose(1, 2).view(batch_size, channels, height, width)

        # Skip connection
        if self.skip_proj is not None:
            residual = self.skip_proj(residual)
        h = h + residual

        return h


class LatentDiffusionDenoiser(nn.Module):
    """
    U-Net-like denoiser for latent diffusion.
    Architecture: 3 down blocks, bottleneck, 3 up blocks.
    Includes cross-attention to spectrum tokens.
    """

    def __init__(
        self,
        latent_channels=1,
        latent_size=16,
        base_channels=64,
        max_channels=256,
        time_emb_dim=128,
        n_attention_heads=4,
        spectrum_token_dim=256,  # From spectrum encoder
        use_spectrum_cross_attention=True,
    ):
        """
        Args:
            latent_channels: Channels in latent space
            latent_size: Size of latent feature maps (e.g., 16x16)
            base_channels: Base number of channels
            max_channels: Maximum number of channels
            time_emb_dim: Time embedding dimension
            n_attention_heads: Number of attention heads
            spectrum_token_dim: Dimension of spectrum tokens
            use_spectrum_cross_attention: Whether to use cross-attention to spectrum
        """
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.base_channels = base_channels
        self.max_channels = max_channels

        # Time embedding
        self.time_embedding = SinusoidalPositionalEmbedding(time_emb_dim)
        spectrum_token_dim = (
            spectrum_token_dim if use_spectrum_cross_attention else None
        )

        self.input_proj = nn.Conv2d(
            latent_channels, base_channels, kernel_size=3, padding=1
        )

        down_channels = [
            base_channels,
            min(base_channels * 2, max_channels),
            min(base_channels * 4, max_channels),
        ]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base_channels
        for idx, out_ch in enumerate(down_channels):
            self.down_blocks.append(
                ResidualBlock(
                    in_ch,
                    out_ch,
                    time_emb_dim,
                    use_attention=(idx == len(down_channels) - 1),
                    n_attention_heads=n_attention_heads,
                    spectrum_token_dim=spectrum_token_dim,
                )
            )
            self.downsamples.append(
                nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            in_ch = out_ch

        bottleneck_ch = min(down_channels[-1] * 2, max_channels)
        self.bottleneck = ResidualBlock(
            down_channels[-1],
            bottleneck_ch,
            time_emb_dim,
            use_attention=True,
            n_attention_heads=n_attention_heads,
            spectrum_token_dim=spectrum_token_dim,
        )

        self.upsamples = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    bottleneck_ch + down_channels[2],
                    down_channels[2],
                    time_emb_dim,
                    use_attention=False,
                    n_attention_heads=n_attention_heads,
                    spectrum_token_dim=spectrum_token_dim,
                ),
                ResidualBlock(
                    down_channels[2] + down_channels[1],
                    down_channels[1],
                    time_emb_dim,
                    use_attention=True,
                    n_attention_heads=n_attention_heads,
                    spectrum_token_dim=spectrum_token_dim,
                ),
                ResidualBlock(
                    down_channels[1] + down_channels[0],
                    down_channels[0],
                    time_emb_dim,
                    use_attention=False,
                    n_attention_heads=n_attention_heads,
                    spectrum_token_dim=spectrum_token_dim,
                ),
            ]
        )

        self.output_proj = nn.Conv2d(base_channels, latent_channels, kernel_size=1)

    def forward(self, x, t, spectrum_tokens=None):
        """
        Args:
            x: (batch_size, latent_channels, latent_size, latent_size)
            t: (batch_size,) time in [0, 1]
            spectrum_tokens: (batch_size, n_tokens, spectrum_token_dim) or None
        Returns:
            out: (batch_size, latent_channels, latent_size, latent_size)
        """
        # Time embedding
        t_emb = self.time_embedding(t)

        h = self.input_proj(x)

        skip_connections = []
        for down_block, downsample in zip(self.down_blocks, self.downsamples):
            h = down_block(h, t_emb, spectrum_tokens)
            skip_connections.append(h)
            h = downsample(h)

        h = self.bottleneck(h, t_emb, spectrum_tokens)

        for upsample, up_block, skip in zip(
            self.upsamples, self.up_blocks, reversed(skip_connections)
        ):
            h = upsample(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = torch.nn.functional.interpolate(
                    h, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, t_emb, spectrum_tokens)

        out = self.output_proj(h)

        return out
