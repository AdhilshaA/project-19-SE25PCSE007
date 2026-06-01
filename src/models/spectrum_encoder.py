"""
Spectrum encoder: converts 1D absorption spectra into embeddings and tokens.
Uses 1D CNN with 3 conv blocks, self-attention, and global pooling.
"""

import torch
import torch.nn as nn


class SpectrumEncoder(nn.Module):
    """
    1D CNN spectrum encoder with attention block.
    Input: (batch_size, 1, n_spectrum_points)
    Output: embedding (batch_size, embed_dim) + token_seq (batch_size, n_tokens, token_dim)
    """

    def __init__(
        self,
        n_spectrum_points=800,
        hidden_dim=64,
        embed_dim=128,
        n_tokens=16,
        n_heads=4,
        use_attention=True,
    ):
        """
        Args:
            n_spectrum_points: Length of input spectrum (default 800)
            hidden_dim: Base hidden dimension (starts at this, doubles with each conv block)
            embed_dim: Final embedding dimension
            n_tokens: Number of attention tokens to output
            n_heads: Number of attention heads
        """
        super().__init__()
        self.n_spectrum_points = n_spectrum_points
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.use_attention = use_attention

        # Three 1D convolution blocks with downsampling
        # Each block: Conv1d -> BatchNorm -> ReLU -> MaxPool
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 800 -> 400
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 400 -> 200
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(
                hidden_dim * 2, hidden_dim * 4, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 200 -> 100
        )

        # Feature dimension after 3 conv blocks
        self.feature_dim = hidden_dim * 4  # 256 if hidden_dim=64
        self.feature_len = 100  # After 3x downsampling: 800 // 8 = 100

        # Self-attention block to create tokens
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=n_heads,
                batch_first=True,
            )
        else:
            self.attention = None

        # Learnable token embeddings (queries for attention)
        self.learnable_tokens = nn.Parameter(torch.randn(1, n_tokens, self.feature_dim))

        # Global average pooling + FC for final embedding
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_embedding = nn.Linear(self.feature_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, 1, n_spectrum_points)
        Returns:
            embedding: (batch_size, embed_dim)
            token_seq: (batch_size, n_tokens, feature_dim)
        """
        # Conv blocks
        x = self.conv_block_1(x)  # (B, hidden_dim, 400)
        x = self.conv_block_2(x)  # (B, hidden_dim*2, 200)
        x = self.conv_block_3(x)  # (B, hidden_dim*4, 100)

        # Transpose for attention (batch_first=True)
        # x is now (B, feature_dim, feature_len)
        x_transposed = x.transpose(1, 2)  # (B, feature_len, feature_dim)

        batch_size = x_transposed.shape[0]
        if self.attention is not None:
            # Self-attention: use learnable tokens as queries
            queries = self.learnable_tokens.expand(
                batch_size, -1, -1
            )  # (B, n_tokens, feature_dim)

            # Apply attention: tokens attend to the spectrum features
            token_seq, _ = self.attention(
                queries, x_transposed, x_transposed
            )  # (B, n_tokens, feature_dim)
        else:
            # CNN-only ablation: reuse pooled convolution features as tokens.
            if x_transposed.shape[1] >= self.n_tokens:
                token_seq = x_transposed[:, : self.n_tokens, :]
            else:
                repeat_count = self.n_tokens - x_transposed.shape[1]
                pad = x_transposed[:, -1:, :].expand(batch_size, repeat_count, -1)
                token_seq = torch.cat([x_transposed, pad], dim=1)

        # Global average pooling on original conv output for embedding
        x_pool = self.global_pool(x).squeeze(-1)  # (B, feature_dim)

        # FC to final embedding dimension
        embedding = self.fc_embedding(x_pool)  # (B, embed_dim)

        return embedding, token_seq


class DesignEncoder(nn.Module):
    """
    Design encoder: encodes geometry image into latent code.
    Simple CNN-based encoder that maps image to a latent vector.
    """

    def __init__(
        self,
        img_size=64,
        in_channels=3,
        hidden_dim=64,
        latent_dim=256,
        latent_channels=1,
        latent_size=16,
    ):
        """
        Args:
            img_size: Input image size (assumed square)
            in_channels: Number of input channels (3 for RGB)
            hidden_dim: Base hidden dimension
            latent_dim: Latent code dimension
        """
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.spatial_latent_dim = latent_channels * latent_size * latent_size

        # Conv blocks to extract features
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

        # After 4 strided convolutions: 64 -> 32 -> 16 -> 8 -> 4
        self.flatten_dim = hidden_dim * 8 * 4 * 4  # 8*hidden_dim*16

        # FC layers to latent code
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def to_spatial(self, latent_code):
        """Reshape a flat latent code into a spatial latent map."""
        if latent_code.dim() != 2:
            raise ValueError("latent_code must have shape (batch_size, latent_dim)")
        if latent_code.shape[1] != self.spatial_latent_dim:
            raise ValueError(
                "latent_dim must match latent_channels * latent_size * latent_size "
                f"({self.spatial_latent_dim}), got {latent_code.shape[1]}"
            )
        return latent_code.view(
            latent_code.shape[0],
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, img_size, img_size)
        Returns:
            latent_code: (batch_size, latent_dim)
        """
        features = self.conv_encoder(x)
        features_flat = features.view(features.shape[0], -1)
        latent_code = self.fc(features_flat)
        return latent_code
