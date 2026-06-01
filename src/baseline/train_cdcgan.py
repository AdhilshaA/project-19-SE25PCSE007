"""
Training script for cDCGAN baseline.
Based on "Global Inverse Design across Multiple Photonic Structure Classes Using Generative Deep Learning"
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_split_dataloaders
from baseline.cdcgan import CDCGANGenerator, CDCGANDiscriminator, init_weights
from utils.training_reports import append_epoch_row, save_curve_plot


def set_seed(manual_seed):
    os.environ["PYTHONHASHSEED"] = str(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_cdcgan(generator, discriminator, dataloader, latent_dim, device):
    generator.eval()
    discriminator.eval()
    criterion = nn.BCELoss()
    total_g_loss = 0.0
    total_d_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        spectra = batch["spectrum"].to(device)
        batch_size_actual = images.shape[0]

        label_real = torch.ones(batch_size_actual, device=device)
        label_fake = torch.zeros(batch_size_actual, device=device)

        d_real = discriminator(images, spectra).view(-1)
        loss_d_real = criterion(d_real, label_real)

        latent_noise = torch.randn(batch_size_actual, latent_dim, device=device)
        fake_images = generator(spectra, latent_noise)
        d_fake = discriminator(fake_images, spectra).view(-1)
        loss_d_fake = criterion(d_fake, label_fake)
        loss_d = loss_d_real + loss_d_fake

        d_fake_for_g = discriminator(fake_images, spectra).view(-1)
        loss_g = criterion(d_fake_for_g, label_real)

        total_g_loss += loss_g.item()
        total_d_loss += loss_d.item()
        n_batches += 1

    return {
        "g_loss": total_g_loss / max(1, n_batches),
        "d_loss": total_d_loss / max(1, n_batches),
    }


def train_cdcgan(
    img_dir,
    spectra_csv,
    output_dir="results/cdcgan",
    run_name=None,
    n_epochs=100,
    batch_size=16,
    learning_rate=0.0001,
    ngf=128,
    ndf=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
    img_size=64,
    n_spectrum_points=800,
    latent_dim=400,
    manual_seed=999,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
    """
    Train cDCGAN baseline model.

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
        latent_dim: Latent noise dimension
        manual_seed: Random seed for reproducibility
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
        ngf=int(ngf),
        ndf=int(ndf),
        img_size=int(img_size),
        n_spectrum_points=int(n_spectrum_points),
        latent_dim=int(latent_dim),
        manual_seed=int(manual_seed),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
    )
    with open(os.path.join(run_dir, "args.json"), "w") as aj:
        json.dump(args_dict, aj, indent=2)

    # Set seed
    set_seed(manual_seed)

    print(f"Random Seed: {manual_seed}")
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {img_dir} and {spectra_csv}")
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
    generator = CDCGANGenerator(
        spectrum_dim=n_spectrum_points,
        latent_dim=latent_dim,
        ngf=ngf,
    ).to(device)
    generator.apply(init_weights)

    discriminator = CDCGANDiscriminator(
        spectrum_dim=n_spectrum_points,
        ndf=ndf,
        image_size=img_size,
    ).to(device)
    discriminator.apply(init_weights)

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(
        f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}"
    )

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )

    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    log_file = os.path.join(run_dir, "training_log.txt")

    g_losses = []
    d_losses = []
    history = []
    csv_file = os.path.join(run_dir, "epoch_metrics.csv")

    with open(log_file, "w") as f:
        f.write(
            f"Split sizes: train={split_sizes['train_size']}, val={split_sizes['val_size']}, test={split_sizes['test_size']}\n"
        )
        for epoch in range(n_epochs):
            generator.train()
            discriminator.train()
            epoch_g_losses = []
            epoch_d_losses = []
            for batch_idx, batch in enumerate(train_loader):
                images = batch["image"].to(device)
                spectra = batch["spectrum"].to(device)
                batch_size_actual = images.shape[0]

                # Real and fake labels
                real_label = random.uniform(0.9, 1.0)
                fake_label = 0.0

                # ============ Train Discriminator ============
                discriminator.zero_grad()

                # Real batch
                label_real = torch.full((batch_size_actual,), real_label, device=device)
                d_real = discriminator(images, spectra).view(-1)
                loss_d_real = criterion(d_real, label_real)
                loss_d_real.backward()
                d_x = d_real.mean().item()

                # Fake batch
                latent_noise = torch.randn(batch_size_actual, latent_dim, device=device)
                fake_images = generator(spectra, latent_noise)

                label_fake = torch.full((batch_size_actual,), fake_label, device=device)
                d_fake = discriminator(fake_images.detach(), spectra).view(-1)
                loss_d_fake = criterion(d_fake, label_fake)
                loss_d_fake.backward()
                d_g_z1 = d_fake.mean().item()

                loss_d = loss_d_real + loss_d_fake
                optimizer_d.step()

                # ============ Train Generator ============
                generator.zero_grad()

                label_real_for_g = torch.full(
                    (batch_size_actual,), real_label, device=device
                )
                d_fake_for_g = discriminator(fake_images, spectra).view(-1)
                loss_g = criterion(d_fake_for_g, label_real_for_g)
                loss_g.backward()
                d_g_z2 = d_fake_for_g.mean().item()
                optimizer_g.step()

                # Log
                if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                    print(
                        f"[{epoch + 1}/{n_epochs}][{batch_idx + 1}/{len(train_loader)}] "
                        f"Loss_D: {loss_d:.4f} Loss_G: {loss_g:.4f} "
                        f"D(x): {d_x:.4f} D(G(z)): {d_g_z1:.4f}/{d_g_z2:.4f}"
                    )

                g_losses.append(loss_g.item())
                d_losses.append(loss_d.item())
                epoch_g_losses.append(loss_g.item())
                epoch_d_losses.append(loss_d.item())

            # End of epoch
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            val_metrics = evaluate_cdcgan(
                generator, discriminator, val_loader, latent_dim, device
            )
            elapsed = time.time() - start_time

            msg = (
                f"Epoch {epoch + 1}/{n_epochs} | Train G Loss: {avg_g_loss:.4f} | "
                f"Train D Loss: {avg_d_loss:.4f} | Val G Loss: {val_metrics['g_loss']:.4f} | "
                f"Val D Loss: {val_metrics['d_loss']:.4f} | Time: {elapsed:.1f}s\n"
            )
            print(msg, end="")
            f.write(msg)

            row = {
                "epoch": epoch + 1,
                "train_g_loss": avg_g_loss,
                "train_d_loss": avg_d_loss,
                "val_g_loss": float(val_metrics["g_loss"]),
                "val_d_loss": float(val_metrics["d_loss"]),
                "elapsed_seconds": elapsed,
            }
            history.append(row)
            append_epoch_row(csv_file, row)

            # Save checkpoints
            if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
                g_path = os.path.join(run_dir, f"generator_epoch_{epoch + 1}.pth")
                d_path = os.path.join(run_dir, f"discriminator_epoch_{epoch + 1}.pth")
                torch.save(generator.state_dict(), g_path)
                torch.save(discriminator.state_dict(), d_path)
                print(f"Checkpoints saved: {g_path}, {d_path}")

    # Save final models
    g_final = os.path.join(run_dir, "generator_final.pth")
    d_final = os.path.join(run_dir, "discriminator_final.pth")
    torch.save(generator.state_dict(), g_final)
    torch.save(discriminator.state_dict(), d_final)

    test_metrics = evaluate_cdcgan(
        generator, discriminator, test_loader, latent_dim, device
    )
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as tf:
        json.dump(test_metrics, tf, indent=2)

    save_curve_plot(
        history,
        series=[("train_g_loss", "train G loss"), ("val_g_loss", "val G loss")],
        output_path=os.path.join(run_dir, "generator_loss_curve.png"),
        title="cDCGAN Generator Loss Curve",
        ylabel="Loss",
    )
    save_curve_plot(
        history,
        series=[("train_d_loss", "train D loss"), ("val_d_loss", "val D loss")],
        output_path=os.path.join(run_dir, "discriminator_loss_curve.png"),
        title="cDCGAN Discriminator Loss Curve",
        ylabel="Loss",
    )

    print(f"\nFinal models saved: {g_final}, {d_final}")

    return generator, discriminator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the cDCGAN baseline")
    parser.add_argument("--img-dir", type=str, default=None)
    parser.add_argument("--spectra-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--manual-seed", type=int, default=999)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--n-spectrum-points", type=int, default=800)
    parser.add_argument("--latent-dim", type=int, default=400)
    parser.add_argument("--ngf", type=int, default=128)
    parser.add_argument("--ndf", type=int, default=64)
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
        else project_root / "results" / "cdcgan"
    )

    if not img_dir.exists():
        print(f"Error: Image directory not found at {img_dir}")
        print("Please extract data/Images.zip first")
        sys.exit(1)

    if not spectra_csv.exists():
        print(f"Error: Spectra CSV not found at {spectra_csv}")
        sys.exit(1)

    # Train
    generator, discriminator = train_cdcgan(
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
        latent_dim=args.latent_dim,
        ngf=args.ngf,
        ndf=args.ndf,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
