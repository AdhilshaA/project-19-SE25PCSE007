"""
Evaluation metrics for inverse design models.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean


class SpectrumMetrics:
    """Compute spectrum-level metrics."""

    @staticmethod
    def mse(y_pred, y_true):
        """Mean squared error."""
        return torch.mean((y_pred - y_true) ** 2)

    @staticmethod
    def mae(y_pred, y_true):
        """Mean absolute error."""
        return torch.mean(torch.abs(y_pred - y_true))

    @staticmethod
    def peak_error(y_pred, y_true):
        """
        Error in absorption peak height.
        Assumes peak is the maximum value in spectrum.
        """
        peak_pred = torch.max(y_pred, dim=-1)[0]
        peak_true = torch.max(y_true, dim=-1)[0]
        return torch.mean(torch.abs(peak_pred - peak_true))

    @staticmethod
    def resonance_shift(y_pred, y_true):
        """
        Error in resonant wavelength (position of peak).
        Assumes peak is the maximum value in spectrum.
        """
        idx_pred = torch.argmax(y_pred, dim=-1).float()
        idx_true = torch.argmax(y_true, dim=-1).float()
        # Convert indices to wavelength shift (assuming linear wavelength spacing)
        shift = torch.mean(torch.abs(idx_pred - idx_true))
        return shift

    @staticmethod
    def cosine_similarity(y_pred, y_true):
        """
        Cosine similarity between predicted and true spectra.
        Returns value in [-1, 1]; 1 is perfect match.
        """
        # Normalize
        y_pred_norm = torch.nn.functional.normalize(y_pred, dim=-1, p=2)
        y_true_norm = torch.nn.functional.normalize(y_true, dim=-1, p=2)
        # Cosine similarity
        sim = torch.sum(y_pred_norm * y_true_norm, dim=-1)
        return torch.mean(sim)

    @staticmethod
    def q_factor_error(y_pred, y_true, fwhm_threshold=0.5):
        """
        Estimate Q-factor from spectrum peaks and estimate error.
        Q ≈ peak_wavelength / FWHM
        """
        # This is a simplified estimate
        # In practice, would need wavelength values to compute actual Q-factor
        peak_pred = torch.max(y_pred, dim=-1)[0]
        peak_true = torch.max(y_true, dim=-1)[0]
        q_error = torch.mean(torch.abs(peak_pred - peak_true))
        return q_error


class DiversityMetrics:
    """Compute diversity metrics for one-to-many sampling."""

    @staticmethod
    def pairwise_distances(samples):
        """
        Compute pairwise L2 distances in sample space.

        Args:
            samples: (n_samples, feature_dim) or (n_samples, C, H, W)
        Returns:
            distances: (n_samples, n_samples)
        """
        if samples.dim() == 4:
            # Image: flatten
            samples_flat = samples.reshape(samples.shape[0], -1)
        else:
            samples_flat = samples

        # Pairwise L2 distance
        distances = torch.cdist(samples_flat, samples_flat, p=2.0)
        return distances

    @staticmethod
    def average_pairwise_distance(samples):
        """Average pairwise distance (measure of diversity)."""
        distances = DiversityMetrics.pairwise_distances(samples)
        # Exclude diagonal (self-distances)
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
        avg_dist = torch.mean(distances[mask])
        return avg_dist

    @staticmethod
    def spectrum_spread(spectra):
        """
        Compute spread in spectrum space for one-to-many samples.
        Metric: variance of spectra across samples.
        """
        spectrum_mean = torch.mean(spectra, dim=0, keepdim=True)
        spectrum_var = torch.mean((spectra - spectrum_mean) ** 2)
        return spectrum_var


class RobustnessMetrics:
    """Compute robustness metrics (e.g., for OOD evaluation)."""

    @staticmethod
    def spec_to_spec_distance(spec1, spec2, metric="mse"):
        """
        Distance between two spectra.

        Args:
            spec1, spec2: (batch_size, n_spectrum_points) or (n_spectrum_points,)
            metric: 'mse', 'mae', or 'cosine'
        Returns:
            distance: scalar or (batch_size,)
        """
        if metric == "mse":
            return torch.mean((spec1 - spec2) ** 2, dim=-1 if spec1.dim() > 1 else None)
        elif metric == "mae":
            return torch.mean(
                torch.abs(spec1 - spec2), dim=-1 if spec1.dim() > 1 else None
            )
        elif metric == "cosine":
            spec1_norm = torch.nn.functional.normalize(spec1, p=2, dim=-1)
            spec2_norm = torch.nn.functional.normalize(spec2, p=2, dim=-1)
            return 1.0 - torch.sum(spec1_norm * spec2_norm, dim=-1)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def success_rate(spectrum_pred, spectrum_true, tolerance=0.1):
        """
        Fraction of samples where predicted spectrum is within tolerance of true.

        Args:
            spectrum_pred: (batch_size, n_spectrum_points)
            spectrum_true: (batch_size, n_spectrum_points)
            tolerance: MSE threshold
        Returns:
            success_rate: scalar in [0, 1]
        """
        mse_per_sample = torch.mean((spectrum_pred - spectrum_true) ** 2, dim=-1)
        successes = (mse_per_sample < tolerance).float()
        return torch.mean(successes)


def evaluate_model(
    model,
    dataloader,
    device="cuda",
    max_batches=None,
):
    """
    Evaluate model on a dataloader.

    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: torch device
        max_batches: Max batches to evaluate (None = all)
    Returns:
        metrics: dict of metric names to values
    """
    model.eval()
    metrics_dict = {
        "mse": [],
        "mae": [],
        "peak_error": [],
        "resonance_shift": [],
        "cosine_similarity": [],
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(device)
            spectra = batch["spectrum"].to(device)

            # Predict spectra
            spectra_pred = model(images)

            # Compute metrics
            metrics_dict["mse"].append(
                SpectrumMetrics.mse(spectra_pred, spectra).item()
            )
            metrics_dict["mae"].append(
                SpectrumMetrics.mae(spectra_pred, spectra).item()
            )
            metrics_dict["peak_error"].append(
                SpectrumMetrics.peak_error(spectra_pred, spectra).item()
            )
            metrics_dict["resonance_shift"].append(
                SpectrumMetrics.resonance_shift(spectra_pred, spectra).item()
            )
            metrics_dict["cosine_similarity"].append(
                SpectrumMetrics.cosine_similarity(spectra_pred, spectra).item()
            )

    # Average metrics
    results = {k: np.mean(v) for k, v in metrics_dict.items()}
    results["std_mse"] = np.std(metrics_dict["mse"])

    return results
