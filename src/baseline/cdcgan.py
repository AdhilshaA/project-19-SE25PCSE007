"""
cDCGAN baseline from the legacy paper.
"Global Inverse Design across Multiple Photonic Structure Classes Using Generative Deep Learning"
https://doi.org/10.1002/adom.202100548

Architecture: Conditional DCGAN with spectrum as condition.
Generator: latent code + spectrum -> geometry image
Discriminator: image + spectrum -> real/fake classification
"""

import torch
import torch.nn as nn


class CDCGANGenerator(nn.Module):
    """
    Conditional DCGAN Generator.
    Input: concatenated spectrum (800 dims) + latent noise (400 dims) reshaped to spatial tensor
    Output: 64x64x3 image
    """

    def __init__(
        self,
        spectrum_dim=800,
        latent_dim=400,
        ngf=128,
    ):
        """
        Args:
            spectrum_dim: Dimension of spectrum (800)
            latent_dim: Dimension of latent noise (400)
            ngf: Base number of generator filters
        """
        super().__init__()
        self.spectrum_dim = spectrum_dim
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.gan_input_dim = spectrum_dim + latent_dim  # 1200

        # Architecture from legacy code
        self.conv1 = nn.ConvTranspose2d(
            self.gan_input_dim, ngf * 8, 6, 1, 0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 6, 2, 4, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 6, 2, 5, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.ConvTranspose2d(ngf, 3, 6, 2, 4, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, spectrum, latent_noise):
        """
        Args:
            spectrum: (batch_size, spectrum_dim)
            latent_noise: (batch_size, latent_dim)
        Returns:
            image: (batch_size, 3, 64, 64)
        """
        # Concatenate spectrum and latent
        combined = torch.cat([spectrum, latent_noise], dim=1)  # (B, 1200)

        # Reshape to spatial tensor (B, 1200, 1, 1)
        x = combined.unsqueeze(-1).unsqueeze(-1)

        # Transposed convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.tanh(x)

        return x


class CDCGANDiscriminator(nn.Module):
    """
    Conditional DCGAN Discriminator.
    Input: image (64x64x3) concatenated with spectrum channels
    Output: 1 (real) or 0 (fake)
    """

    def __init__(
        self,
        spectrum_dim=800,
        ndf=64,
        image_size=64,
    ):
        """
        Args:
            spectrum_dim: Dimension of spectrum (800)
            ndf: Base number of discriminator filters
            image_size: Size of input images (64)
        """
        super().__init__()
        self.spectrum_dim = spectrum_dim
        self.ndf = ndf
        self.image_size = image_size

        # Spectrum to image: convert 800 values to 64x64x3 spatial representation
        self.fc_spectrum = nn.Linear(
            spectrum_dim, image_size * image_size * 3, bias=False
        )

        # Concatenate image (3 channels) + spectrum as image (3 channels) = 6 channels
        self.conv1 = nn.Conv2d(2 * 3, ndf, 6, 2, 4, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 6, 2, 5, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 6, 2, 4, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(ndf * 8, 1, 6, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, spectrum):
        """
        Args:
            image: (batch_size, 3, 64, 64)
            spectrum: (batch_size, spectrum_dim)
        Returns:
            logits: (batch_size, 1)
        """
        batch_size = image.shape[0]

        # Convert spectrum to spatial image representation
        spectrum_spatial = self.fc_spectrum(spectrum)
        spectrum_spatial = spectrum_spatial.view(
            batch_size, 3, self.image_size, self.image_size
        )

        # Concatenate image and spectrum representation (6 channels)
        combined = torch.cat([image, spectrum_spatial], dim=1)

        # Convolutions
        x = self.conv1(combined)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.sigmoid(x)

        return x


def init_weights(m):
    """Initialize weights for generator and discriminator."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
