"""
Command-line entry point for evaluation experiments.
"""

import argparse
import json
import sys
from pathlib import Path

import glob
import os
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloader
from evaluation.metrics import evaluate_model
from evaluation.one_to_many import one_to_many_sampling_test
from evaluation.ood_test import ood_test
from models.forward_predictor import ForwardPredictor
from models.spectrum_encoder import SpectrumEncoder
from models.design_decoder import DesignDecoder
from models.latent_diffusion import LatentDiffusionDenoiser
from models.diffusion_utils import DDPMScheduler
from utils.ema import EMA


def load_state_dict_if_exists(model, checkpoint_path, device):
    if checkpoint_path is None:
        return model
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return (
            value.detach().cpu().item()
            if value.numel() == 1
            else value.detach().cpu().tolist()
        )
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_results(results):
    summary = {}
    for key, value in results.items():
        if key in {"samples", "predicted_spectra"}:
            continue
        if isinstance(value, list):
            summary[key] = [tensor_to_float(item) for item in value]
        else:
            summary[key] = tensor_to_float(value)
    return summary


def build_forward_predictor(args, device):
    model = ForwardPredictor(
        img_size=args.img_size,
        in_channels=3,
        hidden_dim=args.hidden_dim,
        n_spectrum_points=args.n_spectrum_points,
    ).to(device)
    return load_state_dict_if_exists(model, args.forward_predictor_ckpt, device)


def build_inverse_stack(args, device):
    spectrum_encoder = SpectrumEncoder(
        n_spectrum_points=args.n_spectrum_points,
        hidden_dim=args.hidden_dim,
        embed_dim=args.spectrum_embed_dim,
        n_tokens=args.n_tokens,
        n_heads=args.n_heads,
        use_attention=not args.disable_spectrum_attention,
    ).to(device)
    design_decoder = DesignDecoder(
        latent_channels=args.latent_channels,
        latent_size=args.latent_size,
        hidden_dim=args.hidden_dim,
        out_channels=3,
    ).to(device)
    denoiser = LatentDiffusionDenoiser(
        latent_channels=args.latent_channels,
        latent_size=args.latent_size,
        base_channels=args.hidden_dim,
        max_channels=args.max_channels,
        time_emb_dim=args.time_emb_dim,
        n_attention_heads=args.n_heads,
        spectrum_token_dim=args.hidden_dim * 4,
        use_spectrum_cross_attention=not args.disable_cross_attention,
    ).to(device)

    spectrum_encoder = load_state_dict_if_exists(
        spectrum_encoder, args.spectrum_encoder_ckpt, device
    )
    design_decoder = load_state_dict_if_exists(
        design_decoder, args.design_decoder_ckpt, device
    )
    denoiser = load_state_dict_if_exists(denoiser, args.denoiser_ckpt, device)

    # Optionally load EMA checkpoint and copy into denoiser
    if getattr(args, "use_ema", False):
        # Prefer explicit EMA path if provided
        ema_path = getattr(args, "denoiser_ema_ckpt", None)
        if ema_path is None and args.denoiser_ckpt is not None:
            # Look for sibling file in same directory as denoiser_ckpt
            candidate = Path(args.denoiser_ckpt).parent / "denoiser_ema.pth"
            if candidate.exists():
                ema_path = str(candidate)

        if ema_path is not None and Path(ema_path).exists():
            try:
                ema_state = torch.load(ema_path, map_location="cpu")
                ema = EMA(denoiser)
                ema.load_state_dict(ema_state)
                ema.copy_to(denoiser)
                denoiser.to(device)
                print(f"Loaded EMA weights from {ema_path} into denoiser")
            except Exception:
                # Fallback: try to load as a normal model state dict
                try:
                    denoiser.load_state_dict(torch.load(ema_path, map_location=device))
                    denoiser.to(device)
                    print(f"Loaded model-style EMA checkpoint {ema_path} into denoiser")
                except Exception as e:
                    print(f"Warning: failed to load EMA checkpoint {ema_path}: {e}")
        else:
            print(
                "Warning: EMA requested but no EMA checkpoint found; continuing with raw denoiser weights"
            )

    return spectrum_encoder, design_decoder, denoiser


def run_forward(args):
    device = args.device
    if args.forward_predictor_ckpt is None:
        raise ValueError("forward evaluation requires --forward-predictor-ckpt")
    model = build_forward_predictor(args, device)
    dataloader = get_dataloader(
        img_dir=args.img_dir,
        spectra_csv=args.spectra_csv,
        batch_size=args.batch_size,
        num_workers=0,
        image_size=args.img_size,
        n_spectrum_points=args.n_spectrum_points,
        shuffle=False,
        normalize=True,
    )
    results = evaluate_model(
        model, dataloader, device=device, max_batches=args.max_batches
    )
    summary = summarize_results(results)
    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def run_one_to_many(args):
    device = args.device
    if args.forward_predictor_ckpt is None:
        raise ValueError("one-to-many evaluation requires --forward-predictor-ckpt")
    if (
        args.spectrum_encoder_ckpt is None
        or args.design_decoder_ckpt is None
        or args.denoiser_ckpt is None
    ):
        raise ValueError(
            "one-to-many evaluation requires --spectrum-encoder-ckpt, --design-decoder-ckpt, and --denoiser-ckpt"
        )
    forward_predictor = build_forward_predictor(args, device)
    spectrum_encoder, design_decoder, denoiser = build_inverse_stack(args, device)
    scheduler = DDPMScheduler(
        n_steps=args.n_steps, beta_start=args.beta_start, beta_end=args.beta_end
    ).to(device)

    spectra_df = pd.read_csv(args.target_spectra_csv, header=0, index_col=0)
    target_values = spectra_df.iloc[
        args.target_start : args.target_start + args.n_targets, : args.n_spectrum_points
    ].values
    target_spectra = torch.from_numpy(target_values).float()

    results = one_to_many_sampling_test(
        denoiser=denoiser,
        spectrum_encoder=spectrum_encoder,
        design_decoder=design_decoder,
        forward_predictor=forward_predictor,
        scheduler=scheduler,
        target_spectra=target_spectra,
        n_samples=args.n_samples,
        device=device,
        latent_size=args.latent_size,
        latent_channels=args.latent_channels,
        guidance_scale=args.guidance_scale,
        guidance_start_fraction=args.guidance_start_fraction,
        guidance_num_applications=args.guidance_num_applications,
    )
    summary = summarize_results(results)
    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def run_ood(args):
    device = args.device
    if args.forward_predictor_ckpt is None:
        raise ValueError("OOD evaluation requires --forward-predictor-ckpt")
    forward_predictor = build_forward_predictor(args, device)
    ood_dataloader = get_dataloader(
        img_dir=args.img_dir,
        spectra_csv=args.spectra_csv,
        batch_size=args.batch_size,
        num_workers=0,
        image_size=args.img_size,
        n_spectrum_points=args.n_spectrum_points,
        shuffle=False,
        normalize=True,
    )

    results = ood_test(
        model=forward_predictor,
        ood_dataloader=ood_dataloader,
        device=device,
        tolerance=args.tolerance,
    )
    summary = summarize_results(results)
    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run inverse-design evaluation experiments"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--img-dir", type=str, default=None)
    parser.add_argument("--spectra-csv", type=str, default=None)
    parser.add_argument("--target-spectra-csv", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)

    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--n-spectrum-points", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--spectrum-embed-dim", type=int, default=128)
    parser.add_argument("--n-tokens", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--disable-spectrum-attention", action="store_true")

    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=1)
    parser.add_argument("--max-channels", type=int, default=256)
    parser.add_argument("--time-emb-dim", type=int, default=128)
    parser.add_argument("--disable-cross-attention", action="store_true")

    parser.add_argument("--forward-predictor-ckpt", type=str, default=None)
    parser.add_argument("--spectrum-encoder-ckpt", type=str, default=None)
    parser.add_argument("--design-decoder-ckpt", type=str, default=None)
    parser.add_argument("--denoiser-ckpt", type=str, default=None)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Path or glob pattern to a run folder (e.g. results/.../run_*). "
            "If provided, missing checkpoint args will be loaded from this folder. "
            "Individual checkpoint args override files found in the run folder."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    forward_parser = subparsers.add_parser(
        "forward", help="Evaluate the forward predictor"
    )
    forward_parser.add_argument("--max-batches", type=int, default=None)

    one_to_many_parser = subparsers.add_parser(
        "one-to-many", help="Run one-to-many sampling"
    )
    one_to_many_parser.add_argument("--n-targets", type=int, default=1)
    one_to_many_parser.add_argument("--target-start", type=int, default=0)
    one_to_many_parser.add_argument("--n-samples", type=int, default=10)
    one_to_many_parser.add_argument("--n-steps", type=int, default=1000)
    one_to_many_parser.add_argument("--beta-start", type=float, default=1e-4)
    one_to_many_parser.add_argument("--beta-end", type=float, default=2e-2)
    one_to_many_parser.add_argument("--guidance-scale", type=float, default=0.0)
    one_to_many_parser.add_argument(
        "--guidance-start-fraction", type=float, default=0.35
    )
    one_to_many_parser.add_argument(
        "--guidance-num-applications",
        type=int,
        default=None,
        help="Apply guidance only this many times within the late-step window",
    )
    one_to_many_parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Load denoiser_ema.pth into denoiser for sampling/eval",
    )
    one_to_many_parser.add_argument(
        "--denoiser-ema-ckpt",
        type=str,
        default=None,
        help="Explicit path to denoiser_ema.pth (optional)",
    )

    ood_parser = subparsers.add_parser("ood", help="Run OOD evaluation")
    ood_parser.add_argument("--tolerance", type=float, default=0.1)
    ood_parser.add_argument("--n-targets", type=int, default=500)
    ood_parser.add_argument("--target-start", type=int, default=0)
    ood_parser.add_argument("--n-samples", type=int, default=5)
    ood_parser.add_argument("--n-steps", type=int, default=1000)
    ood_parser.add_argument("--beta-start", type=float, default=1e-4)
    ood_parser.add_argument("--beta-end", type=float, default=2e-2)
    ood_parser.add_argument("--guidance-scale", type=float, default=0.0)
    ood_parser.add_argument("--guidance-start-fraction", type=float, default=0.35)
    ood_parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Load denoiser_ema.pth into denoiser for sampling/eval",
    )
    ood_parser.add_argument(
        "--denoiser-ema-ckpt",
        type=str,
        default=None,
        help="Explicit path to denoiser_ema.pth (optional)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # If a run directory or glob pattern is provided, resolve it and populate
    # any missing checkpoint args from files inside that folder. Individual
    # checkpoint CLI args take precedence and will not be overwritten.
    if getattr(args, "run_dir", None) is not None:
        pattern = args.run_dir
        chosen = None
        # If exact path exists and is a directory, prefer it
        if os.path.isdir(pattern):
            chosen = pattern
        else:
            matches = glob.glob(pattern)
            if len(matches) == 0:
                raise FileNotFoundError(
                    f"No run directories matched pattern: {pattern}"
                )
            # If multiple matches, pick the most recently modified
            if len(matches) > 1:
                matches = sorted(matches, key=lambda p: os.path.getmtime(p))
            chosen = matches[-1]

        # normalize to absolute path string
        chosen = os.path.abspath(chosen)
        print(f"Using run dir: {chosen} to fill missing checkpoint args")

        # helper to set ckpt arg if missing and file exists in run dir
        def set_ckpt_if_missing(arg_name, filename):
            if getattr(args, arg_name) is None:
                candidate = os.path.join(chosen, filename)
                if os.path.exists(candidate):
                    setattr(args, arg_name, candidate)

        set_ckpt_if_missing("forward_predictor_ckpt", "forward_predictor.pth")
        set_ckpt_if_missing("spectrum_encoder_ckpt", "spectrum_encoder.pth")
        set_ckpt_if_missing("design_decoder_ckpt", "design_decoder.pth")
        set_ckpt_if_missing("denoiser_ckpt", "denoiser.pth")
        # allow an ema file to be specified in the run dir too
        if getattr(args, "denoiser_ema_ckpt", None) is None:
            ema_candidate = os.path.join(chosen, "denoiser_ema.pth")
            if os.path.exists(ema_candidate):
                args.denoiser_ema_ckpt = ema_candidate

    if args.command == "forward":
        if args.img_dir is None or args.spectra_csv is None:
            raise ValueError("forward evaluation requires --img-dir and --spectra-csv")
        run_forward(args)
    elif args.command == "one-to-many":
        if args.target_spectra_csv is None:
            raise ValueError("one-to-many evaluation requires --target-spectra-csv")
        run_one_to_many(args)
    elif args.command == "ood":
        if args.img_dir is None or args.spectra_csv is None:
            raise ValueError("OOD evaluation requires --img-dir and --spectra-csv")
        run_ood(args)


if __name__ == "__main__":
    main()
