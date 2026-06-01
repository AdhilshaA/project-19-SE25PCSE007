"""
Training script for forward predictor (geometry -> spectrum).
"""

import argparse
import os
import sys
import time
import random
import csv

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
from models.forward_predictor import ForwardPredictor
from evaluation.metrics import evaluate_model
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


def train_forward_predictor(
    img_dir,
    spectra_csv,
    output_dir="results/forward_predictor",
    run_name=None,
    n_epochs=10,
    batch_size=16,
    learning_rate=1e-3,
    hidden_dim=64,
    manual_seed=999,
    device="cuda" if torch.cuda.is_available() else "cpu",
    img_size=64,
    n_spectrum_points=800,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
    """
    Train forward predictor model.

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
    """
    os.makedirs(output_dir, exist_ok=True)
    # Create timestamped run directory
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
        hidden_dim=int(hidden_dim),
        manual_seed=int(manual_seed),
        device=str(device),
        img_size=int(img_size),
        n_spectrum_points=int(n_spectrum_points),
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

    # Create model
    model = ForwardPredictor(
        img_size=img_size,
        in_channels=3,
        hidden_dim=hidden_dim,
        n_spectrum_points=n_spectrum_points,
    ).to(device)

    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
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
            model.train()
            total_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                images = batch["image"].to(device)
                spectra = batch["spectrum"].to(device)

                # Forward pass
                spectrum_pred = model(images)
                loss = criterion(spectrum_pred, spectra)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                    avg_loss = total_loss / n_batches
                    print(
                        f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}, Loss: {avg_loss:.6f}"
                    )

            avg_loss = total_loss / n_batches
            val_metrics = evaluate_model(model, val_loader, device=device)
            elapsed = time.time() - start_time
            msg = (
                f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {avg_loss:.6f} | Val MSE: {val_metrics['mse']:.6f} | "
                f"Time: {elapsed:.1f}s\n"
            )
            print(msg, end="")
            f.write(msg)

            row = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_mse": float(val_metrics["mse"]),
                "val_mae": float(val_metrics["mae"]),
                "val_peak_error": float(val_metrics["peak_error"]),
                "val_resonance_shift": float(val_metrics["resonance_shift"]),
                "val_cosine_similarity": float(val_metrics["cosine_similarity"]),
                "elapsed_seconds": elapsed,
            }
            history.append(row)
            append_epoch_row(csv_file, row)

    # Save model
    model_path = os.path.join(run_dir, "forward_predictor.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Final test evaluation and plots
    test_metrics = evaluate_model(model, test_loader, device=device)
    test_msg = (
        f"\nTest Metrics:\n"
        f"  MSE: {test_metrics['mse']:.6f}\n"
        f"  MAE: {test_metrics['mae']:.6f}\n"
        f"  Peak Error: {test_metrics['peak_error']:.6f}\n"
        f"  Resonance Shift: {test_metrics['resonance_shift']:.6f}\n"
        f"  Cosine Similarity: {test_metrics['cosine_similarity']:.6f}\n"
    )
    print(test_msg)
    with open(log_file, "a") as f:
        f.write(test_msg)

    with open(os.path.join(run_dir, "test_metrics.json"), "w") as tf:
        json.dump(test_metrics, tf, indent=2)

    save_curve_plot(
        history,
        series=[("train_loss", "train loss"), ("val_mse", "val mse")],
        output_path=os.path.join(run_dir, "loss_curve.png"),
        title="Forward Predictor Loss Curve",
        ylabel="Loss",
    )
    save_curve_plot(
        history,
        series=[("val_cosine_similarity", "val cosine similarity")],
        output_path=os.path.join(run_dir, "metric_curve.png"),
        title="Forward Predictor Validation Metric",
        ylabel="Metric",
    )

    return model, model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the forward predictor")
    parser.add_argument("--img-dir", type=str, default=None)
    parser.add_argument("--spectra-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--manual-seed", type=int, default=999)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--n-spectrum-points", type=int, default=800)
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
        else project_root / "results" / "forward_predictor"
    )

    if not img_dir.exists():
        print(f"Error: Image directory not found at {img_dir}")
        print("Please extract data/Images.zip first")
        sys.exit(1)

    if not spectra_csv.exists():
        print(f"Error: Spectra CSV not found at {spectra_csv}")
        sys.exit(1)

    # Train
    model, model_path = train_forward_predictor(
        img_dir=str(img_dir),
        spectra_csv=str(spectra_csv),
        output_dir=str(output_dir),
        run_name=args.run_name,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        manual_seed=args.manual_seed,
        img_size=args.img_size,
        n_spectrum_points=args.n_spectrum_points,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
