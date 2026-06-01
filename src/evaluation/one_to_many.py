"""
One-to-many sampling test: sample multiple diverse geometries for the same spectrum.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import DiversityMetrics, SpectrumMetrics
from models.diffusion_utils import GuidanceHelper


def one_to_many_sampling_test(
    denoiser,
    spectrum_encoder,
    design_decoder,
    forward_predictor,
    scheduler,
    target_spectra,
    n_samples=10,
    device="cuda",
    latent_size=16,
    latent_channels=1,
    guidance_scale=0.0,
    guidance_start_fraction=0.35,
    guidance_num_applications=None,
):
    """
    Sample multiple geometries for the same target spectra.

    Args:
        denoiser: Trained diffusion denoiser
        spectrum_encoder: Spectrum encoder
        design_decoder: Decoder from latent to image space
        forward_predictor: Forward model to verify spectra
        scheduler: Diffusion scheduler
        target_spectra: (n_targets, n_spectrum_points) target spectra
        n_samples: Number of samples per target
        device: torch device
        latent_size: Size of latent spatial features
    Returns:
        results: dict with diversity and accuracy metrics
    """
    denoiser.eval()
    spectrum_encoder.eval()
    if design_decoder is not None:
        design_decoder.eval()
    forward_predictor.eval()

    n_targets = target_spectra.shape[0]
    all_samples = []
    all_spectra_pred = []
    diversity_scores = []
    accuracy_scores = []

    for target_idx in range(n_targets):
        target_spectrum = target_spectra[target_idx : target_idx + 1].to(device)

        with torch.no_grad():
            target_spectrum_reshaped = target_spectrum.unsqueeze(1)
            _, spectrum_tokens = spectrum_encoder(target_spectrum_reshaped)

        spectrum_tokens_batch = spectrum_tokens.expand(n_samples, -1, -1)

        guidance_fn = None
        if guidance_scale > 0:
            guidance_helper = GuidanceHelper(
                forward_predictor=forward_predictor,
                target_spectrum=target_spectrum,
                lambda_guidance=1.0,
            )

            def guidance_fn(x_t, t_step):
                return guidance_helper.compute_guidance(
                    x_t, t_step, decoder=design_decoder
                )

        guidance_start_step = int(scheduler.n_steps * guidance_start_fraction)

        latent_shape = (n_samples, latent_channels, latent_size, latent_size)
        latent = scheduler.sample(
            denoiser,
            latent_shape,
            spectrum_tokens=spectrum_tokens_batch,
            device=device,
            guidance_scale=guidance_scale,
            guidance_fn=guidance_fn,
            guidance_start_step=guidance_start_step,
            guidance_end_step=0,
            guidance_num_applications=guidance_num_applications,
        )

        geometry = design_decoder(latent) if design_decoder is not None else latent

        with torch.no_grad():
            spectrum_pred = forward_predictor(geometry)

        diversity = DiversityMetrics.average_pairwise_distance(geometry)
        diversity_scores.append(diversity.item())

        target_spectrum_expanded = target_spectrum.expand(n_samples, -1)
        accuracy = SpectrumMetrics.cosine_similarity(
            spectrum_pred, target_spectrum_expanded
        )
        accuracy_scores.append(accuracy.item())

        all_samples.append(geometry)
        all_spectra_pred.append(spectrum_pred)

    results = {
        "avg_diversity": sum(diversity_scores) / len(diversity_scores),
        "avg_accuracy": sum(accuracy_scores) / len(accuracy_scores),
        "diversity_per_target": diversity_scores,
        "accuracy_per_target": accuracy_scores,
        "samples": all_samples,
        "predicted_spectra": all_spectra_pred,
    }

    return results
