"""
Demo inference script: load model components from a training `run_dir`, sample designs
for given target spectra, compute evaluation metrics, and save figures and metrics
into a timestamped `results/demo_*` folder.

Usage (example):
 python src/evaluation/demo_inference.py --run-dir results/latent_diffusion/run_20260530-123456 --n-targets 3 --n-samples 4
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.spectrum_encoder import SpectrumEncoder
from models.design_decoder import DesignDecoder
from models.latent_diffusion import LatentDiffusionDenoiser
from models.diffusion_utils import DDPMScheduler
from models.forward_predictor import ForwardPredictor
from evaluation.metrics import SpectrumMetrics, DiversityMetrics
from utils.ema import EMA


def load_run_args(run_dir):
    args_path = os.path.join(run_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            return json.load(f)
    return {}


def safe_load_state(model, path, device="cpu"):
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Demo inference for latent diffusion runs"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--spectra-csv", default=None)
    parser.add_argument("--n-targets", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--n-steps", type=int, default=1000, help="DDPM sampling steps to use"
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_args = load_run_args(run_dir)

    # Create demo output folder
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    demo_dir = os.path.join("results", f"demo_{ts}")
    os.makedirs(demo_dir, exist_ok=True)

    # Save a copy of run args to demo folder
    with open(os.path.join(demo_dir, "run_args.json"), "w") as f:
        json.dump(run_args, f, indent=2)

    device = args.device

    # Defaults (will be overwritten by run_args when available)
    n_spectrum_points = int(run_args.get("n_spectrum_points", 800))
    latent_size = int(run_args.get("latent_size", 16))
    latent_channels = int(run_args.get("latent_channels", 1))
    hidden_dim = int(run_args.get("hidden_dim", 64))
    spectrum_embed_dim = int(run_args.get("spectrum_embed_dim", 128))
    n_tokens = int(run_args.get("n_tokens", 16))
    n_heads = int(run_args.get("n_heads", 4))

    # Instantiate models
    spectrum_encoder = SpectrumEncoder(
        n_spectrum_points=n_spectrum_points,
        hidden_dim=hidden_dim,
        embed_dim=spectrum_embed_dim,
        n_tokens=n_tokens,
        n_heads=n_heads,
        use_attention=run_args.get("spectrum_use_attention", True),
    ).to(device)

    design_decoder = DesignDecoder(
        latent_channels=latent_channels,
        latent_size=latent_size,
        hidden_dim=hidden_dim,
        out_channels=3,
    ).to(device)

    denoiser = LatentDiffusionDenoiser(
        latent_channels=latent_channels,
        latent_size=latent_size,
        base_channels=hidden_dim,
        max_channels=256,
        time_emb_dim=128,
        n_attention_heads=n_heads,
        spectrum_token_dim=hidden_dim * 4,
        use_spectrum_cross_attention=run_args.get("use_spectrum_cross_attention", True),
    ).to(device)

    forward_predictor = ForwardPredictor(
        img_size=64,
        in_channels=3,
        hidden_dim=hidden_dim,
        n_spectrum_points=n_spectrum_points,
    ).to(device)

    # Load checkpoints
    ckpt_paths = {
        "spectrum_encoder": os.path.join(run_dir, "spectrum_encoder.pth"),
        "design_decoder": os.path.join(run_dir, "design_decoder.pth"),
        "denoiser": os.path.join(run_dir, "denoiser.pth"),
        "denoiser_ema": os.path.join(run_dir, "denoiser_ema.pth"),
        "forward_predictor": os.path.join(run_dir, "forward_predictor.pth"),
    }

    if os.path.exists(ckpt_paths["spectrum_encoder"]):
        safe_load_state(spectrum_encoder, ckpt_paths["spectrum_encoder"], device)
    if os.path.exists(ckpt_paths["design_decoder"]):
        safe_load_state(design_decoder, ckpt_paths["design_decoder"], device)
    if os.path.exists(ckpt_paths["denoiser"]):
        safe_load_state(denoiser, ckpt_paths["denoiser"], device)

    # Optionally load EMA and copy values into denoiser
    if args.use_ema and os.path.exists(ckpt_paths["denoiser_ema"]):
        ema = EMA(denoiser)
        ema_state = torch.load(ckpt_paths["denoiser_ema"], map_location="cpu")
        # ema_state might be a state_dict or shadow dict
        try:
            ema.load_state_dict(ema_state)
            ema.copy_to(denoiser)
        except Exception:
            # fallback: load directly into model (if saved as model state)
            denoiser.load_state_dict(ema_state)
        denoiser.to(device)

    # Load forward predictor if available in run folder or default location
    if os.path.exists(ckpt_paths["forward_predictor"]):
        safe_load_state(forward_predictor, ckpt_paths["forward_predictor"], device)
    else:
        # try results/forward_predictor/run_*/forward_predictor.pth
        fp_glob = list(
            Path("results/forward_predictor").glob("**/forward_predictor.pth")
        )
        if fp_glob:
            safe_load_state(forward_predictor, str(fp_glob[-1]), device)

    # Scheduler for sampling (use specified number of steps)
    scheduler = DDPMScheduler(n_steps=args.n_steps)

    # Load target spectra CSV
    spectra_csv = (
        args.spectra_csv
        or run_args.get("spectra_csv")
        or os.path.join("data", "absorptionData_HybridGAN.csv")
    )
    # Match dataset/evaluation CSV parsing and force numeric dtype.
    df = pd.read_csv(spectra_csv, header=0, index_col=0)
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    spectra_all = df_numeric.iloc[:, :n_spectrum_points].to_numpy(dtype=np.float32)
    if np.isnan(spectra_all).any():
        nan_count = int(np.isnan(spectra_all).sum())
        raise ValueError(
            f"Found {nan_count} NaN entries after parsing spectra CSV {spectra_csv}. "
            "Please verify the file has numeric spectrum columns."
        )
    n_available = spectra_all.shape[0]

    rng = np.random.default_rng(0)
    indices = rng.choice(
        n_available, size=min(args.n_targets, n_available), replace=False
    )

    results = []

    for idx in indices:
        target = torch.tensor(spectra_all[idx], dtype=torch.float32, device=device)
        target = target.unsqueeze(0)  # (1, n_points)

        # Encode spectrum tokens
        with torch.no_grad():
            spec_in = target.unsqueeze(1)  # (1,1,n_points)
            _, spec_tokens = spectrum_encoder(spec_in)

        # Prepare batch tokens (repeat for n_samples)
        tokens_batch = spec_tokens.repeat(args.n_samples, 1, 1)

        # Sample latents
        shape = (args.n_samples, latent_channels, latent_size, latent_size)
        with torch.no_grad():
            latents = scheduler.sample(
                denoiser, shape, spectrum_tokens=tokens_batch, device=device
            )

        # Decode to images
        with torch.no_grad():
            images = design_decoder(latents)
            # images in [-1,1] probably; convert to [0,1]
            imgs_np = ((images.clamp(-1, 1) + 1.0) / 2.0).cpu().numpy()

        # Predict spectra of decoded images
        with torch.no_grad():
            spectra_pred = forward_predictor(images.to(device))

        # Compute metrics per sample
        mse_vals = (
            SpectrumMetrics.mse(spectra_pred, target.repeat(args.n_samples, 1))
            .cpu()
            .numpy()
        )
        mae_vals = (
            SpectrumMetrics.mae(spectra_pred, target.repeat(args.n_samples, 1))
            .cpu()
            .numpy()
        )
        cosine_vals = (
            SpectrumMetrics.cosine_similarity(
                spectra_pred, target.repeat(args.n_samples, 1)
            )
            .cpu()
            .numpy()
        )

        avg_mse = float(np.mean(mse_vals))
        avg_mae = float(np.mean(mae_vals))
        avg_cos = float(np.mean(cosine_vals))

        # Diversity
        div = float(DiversityMetrics.average_pairwise_distance(images.cpu()))

        # Create a combined figure: top row images, bottom row spectra overlays
        ncols = args.n_samples
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 6))
        x = np.arange(target.shape[-1])

        for i in range(ncols):
            # Image (top row)
            ax_img = axes[0, i] if ncols > 1 else axes[0]
            img = imgs_np[i].transpose(1, 2, 0)
            ax_img.imshow(img)
            ax_img.axis("off")

            # Spectra (bottom row)
            ax_sp = axes[1, i] if ncols > 1 else axes[1]
            pred = spectra_pred[i].cpu().numpy()
            ax_sp.plot(x, pred, color="C0", alpha=0.8)
            ax_sp.plot(x, target.cpu().numpy().ravel(), color="C1", linewidth=1.5, label="target")
            ax_sp.set_xticks([])
            ax_sp.set_yticks([])

        # Shared title and legend on the bottom-left subplot
        axes[1, 0].legend(loc="upper right") if ncols > 1 else axes[1].legend(loc="upper right")
        fig.suptitle(f"Target idx {idx} — avg_mse {avg_mse:.4f}")
        combined_path = os.path.join(demo_dir, f"combined_idx_{idx}.png")
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        fig.savefig(combined_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        # Also save the separate grid and spectra files for backward compatibility
        img_path = os.path.join(demo_dir, f"samples_idx_{idx}.png")
        sp_path = os.path.join(demo_dir, f"spectra_idx_{idx}.png")
        # write small grid (images only)
        fig_img, axs = plt.subplots(1, args.n_samples, figsize=(3 * args.n_samples, 3))
        for i in range(args.n_samples):
            ax = axs[i] if args.n_samples > 1 else axs
            img = imgs_np[i].transpose(1, 2, 0)
            ax.imshow(img)
            ax.axis("off")
        fig_img.suptitle(f"Target idx {idx} — avg_mse {avg_mse:.4f}")
        fig_img.savefig(img_path, bbox_inches="tight")
        plt.close(fig_img)

        # save spectra-only plot
        fig_sp, axsp = plt.subplots(figsize=(6, 4))
        for i in range(args.n_samples):
            axsp.plot(x, spectra_pred[i].cpu().numpy(), color="C0", alpha=0.6)
        axsp.plot(x, target.cpu().numpy().ravel(), color="C1", linewidth=2, label="target")
        axsp.set_title(f"Spectra for target {idx}")
        axsp.legend()
        fig_sp.savefig(sp_path, bbox_inches="tight")
        plt.close(fig_sp)

        results.append(
            {
                "idx": int(idx),
                "avg_mse": avg_mse,
                "avg_mae": avg_mae,
                "avg_cosine": avg_cos,
                "diversity": div,
                "img_path": img_path,
                "spectra_path": sp_path,
            }
        )

    # Save results JSON
    with open(os.path.join(demo_dir, "metrics.json"), "w") as fj:
        json.dump(results, fj, indent=2)

    print(f"Demo outputs saved to {demo_dir}")


if __name__ == "__main__":
    main()
