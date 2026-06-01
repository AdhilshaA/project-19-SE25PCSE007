"""
Training script for latent diffusion model with spectrum conditioning.
"""

import argparse
import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_split_dataloaders
from models.spectrum_encoder import SpectrumEncoder, DesignEncoder
from models.design_vae import ConvVAEEncoder
from models.design_decoder import DesignDecoder
from models.latent_diffusion import LatentDiffusionDenoiser
from models.diffusion_utils import DDPMScheduler
from utils.ema import EMA
from utils.training_reports import append_epoch_row, save_curve_plot


def set_seed(manual_seed):
    os.environ["PYTHONHASHSEED"] = str(manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_diffusion(
    denoiser,
    spectrum_encoder,
    design_encoder,
    scheduler,
    dataloader,
    device,
    use_vae=False,
    vae_use_mu=True,
):
    denoiser.eval()
    spectrum_encoder.eval()
    design_encoder.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        spectra = batch["spectrum"].to(device)

        spectra_reshaped = spectra.unsqueeze(1)
        _, spectrum_tokens = spectrum_encoder(spectra_reshaped)

        if use_vae:
            mu, logvar = design_encoder(images)
            latent_flat = mu if vae_use_mu else design_encoder.reparameterize(mu, logvar)
            latent_spatial = design_encoder.to_spatial(latent_flat)
        else:
            latent_code = design_encoder(images)
            latent_spatial = design_encoder.to_spatial(latent_code)

        t = torch.rand(images.shape[0], device=device)
        noise = torch.randn_like(latent_spatial)
        noisy_latent = scheduler.add_noise(latent_spatial, t, noise)
        noise_pred = denoiser(noisy_latent, t, spectrum_tokens)
        loss = criterion(noise_pred, noise)

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(1, n_batches)}


def train_latent_diffusion(
    img_dir,
    spectra_csv,
    output_dir="results/latent_diffusion",
    run_name=None,
    n_epochs=5,
    batch_size=8,
    learning_rate=1e-4,
    manual_seed=999,
    device="cuda" if torch.cuda.is_available() else "cpu",
    img_size=64,
    n_spectrum_points=800,
    latent_size=16,
    latent_channels=1,
    hidden_dim=64,
    spectrum_embed_dim=128,
    n_tokens=16,
    n_heads=4,
    spectrum_use_attention=True,
    use_spectrum_cross_attention=True,
    autoencoder_checkpoint_dir=None,
    use_vae=False,
    vae_use_mu=True,
    use_amp=False,
    ema_enabled=True,
    ema_decay=0.9999,
    ema_warmup_steps=0,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
    """
    Train latent diffusion model.

    Args:
        img_dir: Directory with images
        spectra_csv: Path to spectra CSV
        output_dir: Directory to save results
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: torch device
        img_size: Image size
        n_spectrum_points: Spectrum length
        latent_size: Size of latent feature maps
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name_part = (run_name or "").strip().replace(" ", "_")
    run_prefix = f"run_{run_name_part}_" if run_name_part else "run_"
    run_dir = os.path.join(output_dir, f"{run_prefix}{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Save run arguments to JSON for reproducibility
    args_dict = dict(
        img_dir=str(img_dir),
        spectra_csv=str(spectra_csv),
        output_dir=str(output_dir),
        run_dir=str(run_dir),
        run_name=str(run_name) if run_name is not None else None,
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        manual_seed=int(manual_seed),
        device=str(device),
        img_size=int(img_size),
        n_spectrum_points=int(n_spectrum_points),
        latent_size=int(latent_size),
        latent_channels=int(latent_channels),
        hidden_dim=int(hidden_dim),
        spectrum_embed_dim=int(spectrum_embed_dim),
        n_tokens=int(n_tokens),
        n_heads=int(n_heads),
        spectrum_use_attention=bool(spectrum_use_attention),
        use_spectrum_cross_attention=bool(use_spectrum_cross_attention),
        autoencoder_checkpoint_dir=str(autoencoder_checkpoint_dir)
        if autoencoder_checkpoint_dir is not None
        else None,
        use_vae=bool(use_vae),
        vae_use_mu=bool(vae_use_mu),
        use_amp=bool(use_amp),
        ema_enabled=bool(ema_enabled),
        ema_decay=float(ema_decay),
        ema_warmup_steps=int(ema_warmup_steps),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
    )
    with open(os.path.join(run_dir, "args.json"), "w") as aj:
        json.dump(args_dict, aj, indent=2)
    set_seed(manual_seed)

    print(f"Device: {device}")
    print(f"Loading data from {img_dir} and {spectra_csv}")

    # Load data
    train_loader, val_loader, test_loader, split_sizes = get_split_dataloaders(
        img_dir=img_dir,
        spectra_csv=spectra_csv,
        batch_size=batch_size,
        num_workers=0,
        image_size=img_size,
        n_spectrum_points=n_spectrum_points,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        manual_seed=manual_seed,
        normalize=True,
    )

    print(f"Loaded {sum(split_sizes.values())} images")
    print(
        f"Split sizes -> train: {split_sizes['train_size']}, val: {split_sizes['val_size']}, test: {split_sizes['test_size']}"
    )

    # Create models
    print("Creating spectrum encoder...")
    spectrum_encoder = SpectrumEncoder(
        n_spectrum_points=n_spectrum_points,
        hidden_dim=hidden_dim,
        embed_dim=spectrum_embed_dim,
        n_tokens=n_tokens,
        n_heads=n_heads,
        use_attention=spectrum_use_attention,
    ).to(device)

    print("Creating design encoder...")
    latent_dim = latent_channels * latent_size * latent_size
    if use_vae:
        design_encoder = ConvVAEEncoder(
            img_size=img_size,
            in_channels=3,
            hidden_dim=hidden_dim,
            latent_channels=latent_channels,
            latent_size=latent_size,
        ).to(device)
        print("Using Conv-VAE encoder for diffusion (frozen)")
    else:
        design_encoder = DesignEncoder(
            img_size=img_size,
            in_channels=3,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            latent_channels=latent_channels,
            latent_size=latent_size,
        ).to(device)

    print("Creating design decoder...")
    design_decoder = DesignDecoder(
        latent_channels=latent_channels,
        latent_size=latent_size,
        hidden_dim=hidden_dim,
        out_channels=3,
    ).to(device)

    print("Creating diffusion denoiser...")
    denoiser = LatentDiffusionDenoiser(
        latent_channels=latent_channels,
        latent_size=latent_size,
        base_channels=hidden_dim,
        max_channels=256,
        time_emb_dim=128,
        n_attention_heads=n_heads,
        spectrum_token_dim=hidden_dim * 4,
        use_spectrum_cross_attention=use_spectrum_cross_attention,
    ).to(device)

    # Diffusion scheduler
    scheduler = DDPMScheduler(n_steps=1000, beta_start=0.0001, beta_end=0.02).to(device)

    # Count parameters
    total_params = (
        sum(p.numel() for p in spectrum_encoder.parameters())
        + sum(p.numel() for p in design_encoder.parameters())
        + sum(p.numel() for p in design_decoder.parameters())
        + sum(p.numel() for p in denoiser.parameters())
    )
    print(f"Total parameters: {total_params:,}")

    # Optimizer (only train denoiser; encoders are frozen)
    optimizer = optim.Adam(denoiser.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # EMA for denoiser
    ema = None
    global_step = 0
    if ema_enabled:
        ema = EMA(denoiser, decay=ema_decay)

    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    log_file = os.path.join(run_dir, "training_log.txt")

    spectrum_encoder.eval()
    design_encoder.eval()
    design_decoder.eval()

    for module in (spectrum_encoder, design_encoder, design_decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    if autoencoder_checkpoint_dir is not None:
        encoder_path = Path(autoencoder_checkpoint_dir) / "design_encoder.pth"
        decoder_path = Path(autoencoder_checkpoint_dir) / "design_decoder.pth"
        if encoder_path.exists():
            try:
                design_encoder.load_state_dict(
                    torch.load(encoder_path, map_location=device)
                )
            except Exception:
                # If VAE encoder has different keys, attempt partial load
                state = torch.load(encoder_path, map_location=device)
                design_encoder.load_state_dict(state, strict=False)
        else:
            print(f"Warning: missing design encoder checkpoint at {encoder_path}")
        if decoder_path.exists():
            design_decoder.load_state_dict(
                torch.load(decoder_path, map_location=device)
            )
        else:
            print(f"Warning: missing design decoder checkpoint at {decoder_path}")
    else:
        print(
            "Warning: no autoencoder checkpoint directory provided; diffusion will use an uninitialized latent geometry stack."
        )

    with open(log_file, "w") as f:
        # Header with run info
        f.write(f"Run args written to {os.path.join(run_dir, 'args.json')}\n")
        f.write(
            f"EMA enabled: {ema_enabled}, decay: {ema_decay}, warmup_steps: {ema_warmup_steps}\n"
        )
        f.write(
            f"Split sizes: train={split_sizes['train_size']}, val={split_sizes['val_size']}, test={split_sizes['test_size']}\n"
        )
        csv_file = os.path.join(run_dir, "epoch_metrics.csv")
        history = []
        for epoch in range(n_epochs):
            denoiser.train()
            total_loss = 0.0
            n_batches = 0
            scaler = (
                torch.cuda.amp.GradScaler()
                if (use_amp and device.startswith("cuda"))
                else None
            )

            for batch_idx, batch in enumerate(train_loader):
                images = batch["image"].to(device)
                spectra = batch["spectrum"].to(device)

                # Encode spectrum to tokens
                with torch.no_grad():
                    spectra_reshaped = spectra.unsqueeze(1)  # (B, 1, n_spectrum_points)
                    spectrum_embed, spectrum_tokens = spectrum_encoder(spectra_reshaped)

                    # Encode geometry to latent code (VAE encoder returns mu,logvar)
                    if use_vae:
                        mu, logvar = design_encoder(images)
                        if vae_use_mu:
                            latent_flat = mu
                        else:
                            # sample from posterior
                            latent_flat = design_encoder.reparameterize(mu, logvar)
                        latent_spatial = design_encoder.to_spatial(latent_flat)
                    else:
                        latent_code = design_encoder(images)
                        latent_spatial = design_encoder.to_spatial(latent_code)

                # Sample random timesteps
                t = torch.rand(images.shape[0], device=device)

                # Add noise
                noise = torch.randn_like(latent_spatial)
                noisy_latent = scheduler.add_noise(latent_spatial, t, noise)

                # Denoise prediction
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        noise_pred = denoiser(noisy_latent, t, spectrum_tokens)
                        loss = criterion(noise_pred, noise)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                    if ema is not None and global_step > ema_warmup_steps:
                        ema.update(denoiser)
                else:
                    noise_pred = denoiser(noisy_latent, t, spectrum_tokens)
                    loss = criterion(noise_pred, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                    optimizer.step()
                    global_step += 1
                    if ema is not None and global_step > ema_warmup_steps:
                        ema.update(denoiser)

                total_loss += loss.item()
                n_batches += 1

                if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                    avg_loss = total_loss / n_batches
                    print(
                        f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}, Loss: {avg_loss:.6f}"
                    )

            avg_loss = total_loss / n_batches
            val_metrics = evaluate_diffusion(
                denoiser,
                spectrum_encoder,
                design_encoder,
                scheduler,
                val_loader,
                device=device,
                use_vae=use_vae,
                vae_use_mu=vae_use_mu,
            )
            elapsed = time.time() - start_time
            msg = (
                f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_metrics['loss']:.6f} | "
                f"Time: {elapsed:.1f}s\n"
            )
            print(msg, end="")
            f.write(msg)

            row = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": float(val_metrics["loss"]),
                "elapsed_seconds": elapsed,
            }
            history.append(row)
            append_epoch_row(csv_file, row)

    # Save models
    spectrum_encoder_path = os.path.join(run_dir, "spectrum_encoder.pth")
    design_encoder_path = os.path.join(run_dir, "design_encoder.pth")
    design_decoder_path = os.path.join(run_dir, "design_decoder.pth")
    denoiser_path = os.path.join(run_dir, "denoiser.pth")

    torch.save(spectrum_encoder.state_dict(), spectrum_encoder_path)
    torch.save(design_encoder.state_dict(), design_encoder_path)
    torch.save(design_decoder.state_dict(), design_decoder_path)
    torch.save(denoiser.state_dict(), denoiser_path)
    # Save EMA checkpoint if available
    if ema is not None:
        denoiser_ema_path = os.path.join(run_dir, "denoiser_ema.pth")
        ema_sd = ema.as_state_dict_for_saving(denoiser)
        torch.save(ema_sd, denoiser_ema_path)
        print(f"  {denoiser_ema_path}")

    print(f"\nModels saved:")
    print(f"  {spectrum_encoder_path}")
    print(f"  {design_encoder_path}")
    print(f"  {design_decoder_path}")
    print(f"  {denoiser_path}")

    test_metrics = evaluate_diffusion(
        denoiser,
        spectrum_encoder,
        design_encoder,
        scheduler,
        test_loader,
        device=device,
        use_vae=use_vae,
        vae_use_mu=vae_use_mu,
    )
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as tf:
        json.dump(test_metrics, tf, indent=2)

    save_curve_plot(
        history,
        series=[("train_loss", "train loss"), ("val_loss", "val loss")],
        output_path=os.path.join(run_dir, "loss_curve.png"),
        title="Diffusion Loss Curve",
        ylabel="Loss",
    )

    return spectrum_encoder, design_encoder, design_decoder, denoiser, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the latent diffusion model")
    parser.add_argument("--img-dir", type=str, default=None)
    parser.add_argument("--spectra-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--manual-seed", type=int, default=999)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--n-spectrum-points", type=int, default=800)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--spectrum-embed-dim", type=int, default=128)
    parser.add_argument("--n-tokens", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--disable-spectrum-attention", action="store_true")
    parser.add_argument("--disable-cross-attention", action="store_true")
    parser.add_argument(
        "--use-vae",
        action="store_true",
        help="Use Conv-VAE encoder for diffusion (load VAE encoder/decoder)",
    )
    veamut = parser.add_mutually_exclusive_group()
    veamut.add_argument(
        "--vae-use-mu",
        dest="vae_use_mu",
        action="store_true",
        help="When using VAE, use the encoder mu as the latent (default)",
    )
    veamut.add_argument(
        "--no-vae-use-mu",
        dest="vae_use_mu",
        action="store_false",
        help="When using VAE, sample from posterior instead of using mu",
    )
    parser.set_defaults(vae_use_mu=True)
    parser.add_argument("--autoencoder-checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--use-amp", action="store_true", help="Enable AMP (mixed precision)"
    )
    parser.add_argument(
        "--disable-ema", action="store_true", help="Disable EMA for denoiser"
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.9999, help="EMA decay for denoiser"
    )
    parser.add_argument(
        "--ema-warmup-steps",
        type=int,
        default=0,
        help="Steps to wait before starting EMA updates",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent.parent
    img_dir = (
        Path(args.img_dir)
        if args.img_dir is not None
        else project_root / "data" / "Images"
    )
    spectra_csv = (
        Path(args.spectra_csv)
        if args.spectra_csv is not None
        else project_root / "data" / "absorptionData_HybridGAN.csv"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else project_root / "results" / "latent_diffusion"
    )

    if not img_dir.exists():
        print(f"Error: Image directory not found at {img_dir}")
        print("Please extract data/Images.zip first")
        sys.exit(1)

    if not spectra_csv.exists():
        print(f"Error: Spectra CSV not found at {spectra_csv}")
        sys.exit(1)

    # Train
    spectrum_encoder, design_encoder, design_decoder, denoiser, scheduler = (
        train_latent_diffusion(
            img_dir=str(img_dir),
            spectra_csv=str(spectra_csv),
            output_dir=str(output_dir),
            run_name=args.run_name,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            manual_seed=args.manual_seed,
            img_size=args.img_size,
            n_spectrum_points=args.n_spectrum_points,
            latent_size=args.latent_size,
            latent_channels=args.latent_channels,
            hidden_dim=args.hidden_dim,
            spectrum_embed_dim=args.spectrum_embed_dim,
            n_tokens=args.n_tokens,
            n_heads=args.n_heads,
            spectrum_use_attention=not args.disable_spectrum_attention,
            use_spectrum_cross_attention=not args.disable_cross_attention,
            autoencoder_checkpoint_dir=args.autoencoder_checkpoint_dir,
            use_amp=args.use_amp,
            use_vae=args.use_vae,
            vae_use_mu=args.vae_use_mu,
            ema_enabled=not args.disable_ema,
            ema_decay=args.ema_decay,
            ema_warmup_steps=args.ema_warmup_steps,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    )
