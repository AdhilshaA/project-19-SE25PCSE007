"""
Out-of-distribution (OOD) test: evaluate model on held-out spectral shapes.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import SpectrumMetrics, RobustnessMetrics


def ood_test(
    model,
    ood_dataloader,
    device="cuda",
    tolerance=0.1,
):
    """
    Evaluate model on OOD test set.

    Args:
        model: Forward predictor or full pipeline
        ood_dataloader: DataLoader with OOD samples
        device: torch device
        tolerance: MSE threshold for success
    Returns:
        results: dict with OOD metrics
    """
    model.eval()

    mse_list = []
    mae_list = []
    peak_error_list = []
    cosine_sim_list = []

    with torch.no_grad():
        for batch in ood_dataloader:
            images = batch["image"].to(device)
            spectra = batch["spectrum"].to(device)

            # Predict spectra
            spectra_pred = model(images)

            # Compute metrics
            mse_list.append(SpectrumMetrics.mse(spectra_pred, spectra).item())
            mae_list.append(SpectrumMetrics.mae(spectra_pred, spectra).item())
            peak_error_list.append(
                SpectrumMetrics.peak_error(spectra_pred, spectra).item()
            )
            cosine_sim_list.append(
                SpectrumMetrics.cosine_similarity(spectra_pred, spectra).item()
            )

    results = {
        "ood_mse": np.mean(mse_list),
        "ood_mae": np.mean(mae_list),
        "ood_peak_error": np.mean(peak_error_list),
        "ood_cosine_similarity": np.mean(cosine_sim_list),
        "ood_std_mse": np.std(mse_list),
    }

    return results


def compare_ood_vs_iid(
    model,
    iid_dataloader,
    ood_dataloader,
    device="cuda",
):
    """
    Compare model performance on IID vs OOD samples.

    Args:
        model: Model to evaluate
        iid_dataloader: IID validation set
        ood_dataloader: OOD test set
        device: torch device
    Returns:
        comparison: dict with IID and OOD results
    """
    # Evaluate on IID
    iid_results = ood_test(model, iid_dataloader, device)

    # Evaluate on OOD
    ood_results = ood_test(model, ood_dataloader, device)

    # Compute degradation
    degradation = {
        "mse_increase": (ood_results["ood_mse"] - iid_results["ood_mse"])
        / iid_results["ood_mse"],
        "similarity_decrease": (
            iid_results["ood_cosine_similarity"] - ood_results["ood_cosine_similarity"]
        ),
    }

    comparison = {
        "iid": iid_results,
        "ood": ood_results,
        "degradation": degradation,
    }

    return comparison
