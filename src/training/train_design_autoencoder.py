"""
Training script for the design autoencoder (geometry -> latent -> geometry).
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_split_dataloaders
from models.spectrum_encoder import DesignEncoder
from models.design_decoder import DesignDecoder, DesignAutoencoder
from models.design_vae import ConvVAEEncoder, ConvVAE
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
def evaluate_autoencoder(
    autoencoder,
    dataloader,
    device,
    use_vae=False,
    beta=1.0,
    kl_warmup_epochs=5,
    current_epoch=1,
    kl_free_bits=0.0,
    kl_normalize_by_dim=True,
):
    autoencoder.eval()
    recon_criterion = nn.L1Loss()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        if use_vae:
            recon_images, mu, logvar, _ = autoencoder(images, sample=False)
            recon_loss = recon_criterion(recon_images, images)
            kl = autoencoder.kl_loss(
                mu,
                logvar,
                free_bits=kl_free_bits,
                normalize_by_dim=kl_normalize_by_dim,
            )
            kl_weight = beta * min(1.0, current_epoch / max(1, kl_warmup_epochs))
            loss = recon_loss + kl_weight * kl
            total_kl += kl.item()
        else:
            recon_images = autoencoder(images)
            recon_loss = recon_criterion(recon_images, images)
            loss = recon_loss

        total_loss += loss.item()
        total_recon += recon_loss.item()
        n_batches += 1

    if n_batches == 0:
        return {
            "loss": float("nan"),
            "recon_loss": float("nan"),
            "kl_loss": float("nan"),
        }

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches if use_vae else 0.0,
    }


def train_design_autoencoder(
    img_dir,
    spectra_csv,
    output_dir="results/design_autoencoder",
    run_name=None,
    n_epochs=20,
    batch_size=16,
    learning_rate=1e-4,
    manual_seed=999,
    device="cuda" if torch.cuda.is_available() else "cpu",
    img_size=64,
    latent_channels=1,
    latent_size=16,
    hidden_dim=64,
    use_vae=False,
    beta=0.05,
    kl_warmup_epochs=20,
    kl_free_bits=0.03,
    kl_normalize_by_dim=True,
    use_amp=False,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
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
        latent_channels=int(latent_channels),
        latent_size=int(latent_size),
        hidden_dim=int(hidden_dim),
        use_vae=bool(use_vae),
        beta=float(beta),
        kl_warmup_epochs=int(kl_warmup_epochs),
        kl_free_bits=float(kl_free_bits),
        kl_normalize_by_dim=bool(kl_normalize_by_dim),
        use_amp=bool(use_amp),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
    )
    with open(os.path.join(run_dir, "args.json"), "w") as aj:
        json.dump(args_dict, aj, indent=2)
    set_seed(manual_seed)

    print(f"Device: {device}")
    print(f"Loading data from {img_dir} and {spectra_csv}")

    train_loader, val_loader, test_loader, split_sizes = get_split_dataloaders(
        img_dir=img_dir,
        spectra_csv=spectra_csv,
        batch_size=batch_size,
        num_workers=0,
        image_size=img_size,
        n_spectrum_points=800,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        manual_seed=manual_seed,
        normalize=True,
    )

    print(
        f"Split sizes -> train: {split_sizes['train_size']}, val: {split_sizes['val_size']}, test: {split_sizes['test_size']}"
    )

    latent_dim = latent_channels * latent_size * latent_size

    # Create encoder/decoder or VAE
    decoder = DesignDecoder(
        latent_channels=latent_channels,
        latent_size=latent_size,
        hidden_dim=hidden_dim,
        out_channels=3,
    ).to(device)

    if use_vae:
        encoder = ConvVAEEncoder(
            img_size=img_size,
            in_channels=3,
            hidden_dim=hidden_dim,
            latent_channels=latent_channels,
            latent_size=latent_size,
        ).to(device)
        autoencoder = ConvVAE(encoder, decoder).to(device)
        print("Training as Conv-VAE")
    else:
        encoder = DesignEncoder(
            img_size=img_size,
            in_channels=3,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            latent_channels=latent_channels,
            latent_size=latent_size,
        ).to(device)
        autoencoder = DesignAutoencoder(encoder, decoder).to(device)

    # Optimizer: AdamW for better weight decay handling
    optimizer = optim.AdamW(
        autoencoder.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    recon_criterion = nn.L1Loss()

    print(
        f"Autoencoder parameters: {sum(p.numel() for p in autoencoder.parameters()):,}"
    )
    print(
        f"Use VAE: {use_vae}, AMP: {use_amp}, beta: {beta}, "
        f"kl_warmup_epochs: {kl_warmup_epochs}, kl_free_bits: {kl_free_bits}, "
        f"kl_normalize_by_dim: {kl_normalize_by_dim}"
    )
    print("\nStarting training...")
    start_time = time.time()
    log_file = os.path.join(run_dir, "training_log.txt")
    csv_file = os.path.join(run_dir, "epoch_metrics.csv")
    history = []

    with open(log_file, "w") as f:
        f.write(
            f"Split sizes: train={split_sizes['train_size']}, val={split_sizes['val_size']}, test={split_sizes['test_size']}\n"
        )
        for epoch in range(n_epochs):
            autoencoder.train()
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            n_batches = 0

            scaler = (
                torch.cuda.amp.GradScaler()
                if (use_amp and device.startswith("cuda"))
                else None
            )

            for batch_idx, batch in enumerate(train_loader):
                images = batch["image"].to(device)

                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        if use_vae:
                            recon_images, mu, logvar, _ = autoencoder(
                                images, sample=True
                            )
                            recon_loss = recon_criterion(recon_images, images)
                            kl = autoencoder.kl_loss(
                                mu,
                                logvar,
                                free_bits=kl_free_bits,
                                normalize_by_dim=kl_normalize_by_dim,
                            )
                            # KL warmup
                            curr_epoch = epoch + 1
                            kl_weight = beta * min(
                                1.0, curr_epoch / max(1, kl_warmup_epochs)
                            )
                            loss = recon_loss + kl_weight * kl
                            recon_value = recon_loss.item()
                            kl_value = kl.item()
                        else:
                            recon_images = autoencoder(images)
                            loss = recon_criterion(recon_images, images)
                            recon_value = loss.item()
                            kl_value = 0.0

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if use_vae:
                        recon_images, mu, logvar, _ = autoencoder(images, sample=True)
                        recon_loss = recon_criterion(recon_images, images)
                        kl = autoencoder.kl_loss(
                            mu,
                            logvar,
                            free_bits=kl_free_bits,
                            normalize_by_dim=kl_normalize_by_dim,
                        )
                        curr_epoch = epoch + 1
                        kl_weight = beta * min(
                            1.0, curr_epoch / max(1, kl_warmup_epochs)
                        )
                        loss = recon_loss + kl_weight * kl
                        recon_value = recon_loss.item()
                        kl_value = kl.item()
                    else:
                        recon_images = autoencoder(images)
                        loss = recon_criterion(recon_images, images)
                        recon_value = loss.item()
                        kl_value = 0.0

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                total_recon += recon_value
                total_kl += kl_value
                n_batches += 1

                if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                    avg_loss = total_loss / n_batches
                    print(
                        f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}, Loss: {avg_loss:.6f}"
                    )

            avg_loss = total_loss / n_batches
            avg_recon = total_recon / n_batches
            avg_kl = total_kl / n_batches if use_vae else 0.0
            val_metrics = evaluate_autoencoder(
                autoencoder,
                val_loader,
                device=device,
                use_vae=use_vae,
                beta=beta,
                kl_warmup_epochs=kl_warmup_epochs,
                current_epoch=epoch + 1,
                kl_free_bits=kl_free_bits,
                kl_normalize_by_dim=kl_normalize_by_dim,
            )
            elapsed = time.time() - start_time
            msg = (
                f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {avg_loss:.6f} | "
                f"Val Loss: {val_metrics['loss']:.6f} | Time: {elapsed:.1f}s\n"
            )
            print(msg, end="")
            f.write(msg)

            row = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_recon_loss": avg_recon,
                "train_kl_loss": avg_kl,
                "val_loss": float(val_metrics["loss"]),
                "val_recon_loss": float(val_metrics["recon_loss"]),
                "val_kl_loss": float(val_metrics["kl_loss"]),
                "elapsed_seconds": elapsed,
            }
            history.append(row)
            append_epoch_row(csv_file, row)

    encoder_path = os.path.join(run_dir, "design_encoder.pth")
    decoder_path = os.path.join(run_dir, "design_decoder.pth")
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

    test_metrics = evaluate_autoencoder(
        autoencoder,
        test_loader,
        device=device,
        use_vae=use_vae,
        beta=beta,
        kl_warmup_epochs=kl_warmup_epochs,
        current_epoch=n_epochs,
        kl_free_bits=kl_free_bits,
        kl_normalize_by_dim=kl_normalize_by_dim,
    )
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as tf:
        json.dump(test_metrics, tf, indent=2)

    save_curve_plot(
        history,
        series=[("train_loss", "train loss"), ("val_loss", "val loss")],
        output_path=os.path.join(run_dir, "loss_curve.png"),
        title="Autoencoder Loss Curve",
        ylabel="Loss",
    )

    print(f"\nModels saved:\n  {encoder_path}\n  {decoder_path}")
    return encoder, decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the design autoencoder")
    parser.add_argument("--img-dir", type=str, default=None)
    parser.add_argument("--spectra-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--manual-seed", type=int, default=999)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--vae",
        action="store_true",
        help="Train a Conv-VAE instead of deterministic AE",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="KL weight for VAE (smaller default to reduce posterior collapse)",
    )
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=20,
        help="KL warmup epochs",
    )
    parser.add_argument(
        "--kl-free-bits",
        type=float,
        default=0.03,
        help="Free-bits threshold per latent dim in KL term",
    )
    knorm = parser.add_mutually_exclusive_group()
    knorm.add_argument(
        "--kl-normalize-by-dim",
        dest="kl_normalize_by_dim",
        action="store_true",
        help="Average KL over latent dimensions (recommended)",
    )
    knorm.add_argument(
        "--no-kl-normalize-by-dim",
        dest="kl_normalize_by_dim",
        action="store_false",
        help="Sum KL over latent dimensions",
    )
    parser.set_defaults(kl_normalize_by_dim=True)
    parser.add_argument(
        "--use-amp", action="store_true", help="Use mixed precision (AMP)"
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

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
        else project_root / "results" / "design_autoencoder"
    )

    if not img_dir.exists():
        print(f"Error: Image directory not found at {img_dir}")
        sys.exit(1)
    if not spectra_csv.exists():
        print(f"Error: Spectra CSV not found at {spectra_csv}")
        sys.exit(1)

    train_design_autoencoder(
        img_dir=str(img_dir),
        spectra_csv=str(spectra_csv),
        output_dir=str(output_dir),
        run_name=args.run_name,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        manual_seed=args.manual_seed,
        img_size=args.img_size,
        latent_channels=args.latent_channels,
        latent_size=args.latent_size,
        hidden_dim=args.hidden_dim,
        use_vae=args.vae,
        beta=args.beta,
        kl_warmup_epochs=args.kl_warmup_epochs,
        kl_free_bits=args.kl_free_bits,
        kl_normalize_by_dim=args.kl_normalize_by_dim,
        use_amp=args.use_amp,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
