#!/usr/bin/env bash
set -euo pipefail

# Minimal-time smoke test for full pipeline integrity.
# Goal: catch wiring errors quickly (checkpoints, logging, JSON outputs, figures).
#
# Run from project root:
#   bash scripts/all_experiments_smoke.sh
#
# Optional: mid-length stability run (~4-5h, hardware-dependent)
#   STABILITY_TEST=1 bash scripts/all_experiments_smoke.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
IMG_DIR="${IMG_DIR:-data/Images}"
SPECTRA_CSV="${SPECTRA_CSV:-data/absorptionData_HybridGAN.csv}"
MANUAL_SEED="${MANUAL_SEED:-999}"
RUN_NAME="${RUN_NAME:-smoke}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
# Optional dedicated OOD dataset (leave empty to skip OOD check)
OOD_IMG_DIR="${OOD_IMG_DIR:-}"
OOD_SPECTRA_CSV="${OOD_SPECTRA_CSV:-}"

BATCH_SIZE="${BATCH_SIZE:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
LATENT_SIZE="${LATENT_SIZE:-16}"
LATENT_CHANNELS="${LATENT_CHANNELS:-1}"
N_SPECTRUM_POINTS="${N_SPECTRUM_POINTS:-800}"

# Very short smoke epochs
FP_EPOCHS="${FP_EPOCHS:-1}"
AE_EPOCHS="${AE_EPOCHS:-1}"
DIFF_EPOCHS="${DIFF_EPOCHS:-1}"
CDCGAN_EPOCHS="${CDCGAN_EPOCHS:-1}"

# Optional mid-length stability epochs (4-5h target depends on GPU)
STABILITY_TEST="${STABILITY_TEST:-0}"
FP_EPOCHS_STAB="${FP_EPOCHS_STAB:-5}"
AE_EPOCHS_STAB="${AE_EPOCHS_STAB:-8}"
DIFF_EPOCHS_STAB="${DIFF_EPOCHS_STAB:-8}"
CDCGAN_EPOCHS_STAB="${CDCGAN_EPOCHS_STAB:-30}"

LOG_DIR="results/experiment_logs"
mkdir -p "$LOG_DIR"
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/all_experiments_smoke_${RUN_TAG}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

latest_run_dir() {
  local base_dir="$1"
  if [[ ! -d "$base_dir" ]]; then
    echo ""; return 1
  fi
  local latest
  latest="$(ls -dt "$base_dir"/run_* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo ""; return 1
  fi
  echo "$latest"
}

run_cmd() {
  echo
  echo "[CMD] $*"
  "$@"
}

check_required_paths() {
  if [[ ! -d "$IMG_DIR" ]]; then
    echo "Error: Missing image directory: $IMG_DIR"
    exit 1
  fi
  if [[ ! -f "$SPECTRA_CSV" ]]; then
    echo "Error: Missing spectra CSV: $SPECTRA_CSV"
    exit 1
  fi
}

resolve_epochs() {
  if [[ "$STABILITY_TEST" == "1" ]]; then
    FP_E="$FP_EPOCHS_STAB"
    AE_E="$AE_EPOCHS_STAB"
    DIFF_E="$DIFF_EPOCHS_STAB"
    CDCGAN_E="$CDCGAN_EPOCHS_STAB"
    echo "Stability test mode enabled."
  else
    FP_E="$FP_EPOCHS"
    AE_E="$AE_EPOCHS"
    DIFF_E="$DIFF_EPOCHS"
    CDCGAN_E="$CDCGAN_EPOCHS"
    echo "Smoke mode enabled."
  fi
}

echo "============================================================"
echo "SMOKE / STABILITY PIPELINE"
echo "Log file: $LOG_FILE"
echo "============================================================"

check_required_paths
resolve_epochs

# 1) Forward predictor
run_cmd "$PYTHON_BIN" src/training/train_forward_predictor.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results_smoke/forward_predictor \
  --epochs "$FP_E" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --n-spectrum-points "$N_SPECTRUM_POINTS" \
  --run-name "$RUN_NAME" \
  --manual-seed "$MANUAL_SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"
FP_RUN="$(latest_run_dir results_smoke/forward_predictor)"

# 2) Deterministic AE
run_cmd "$PYTHON_BIN" src/training/train_design_autoencoder.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results_smoke/design_autoencoder_det \
  --epochs "$AE_E" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --latent-size "$LATENT_SIZE" \
  --latent-channels "$LATENT_CHANNELS" \
  --run-name "$RUN_NAME" \
  --manual-seed "$MANUAL_SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"
AE_RUN="$(latest_run_dir results_smoke/design_autoencoder_det)"

# 3) VAE AE
run_cmd "$PYTHON_BIN" src/training/train_design_autoencoder.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results_smoke/design_autoencoder_vae \
  --epochs "$AE_E" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --latent-size "$LATENT_SIZE" \
  --latent-channels "$LATENT_CHANNELS" \
  --run-name "$RUN_NAME" \
  --manual-seed "$MANUAL_SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO" \
  --vae \
  --beta 1.0 \
  --kl-warmup-epochs 2
AE_RUN_VAE="$(latest_run_dir results_smoke/design_autoencoder_vae)"

# 4) Diffusion
run_cmd "$PYTHON_BIN" src/training/train_diffusion.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results_smoke/latent_diffusion \
  --epochs "$DIFF_E" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --latent-size "$LATENT_SIZE" \
  --latent-channels "$LATENT_CHANNELS" \
  --n-spectrum-points "$N_SPECTRUM_POINTS" \
  --autoencoder-checkpoint-dir "$AE_RUN" \
  --run-name "$RUN_NAME" \
  --manual-seed "$MANUAL_SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"
DIFF_RUN="$(latest_run_dir results_smoke/latent_diffusion)"

# 5) VAE diffusion
run_cmd "$PYTHON_BIN" src/training/train_diffusion.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results_smoke/latent_diffusion_vae \
  --epochs "$DIFF_E" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --latent-size "$LATENT_SIZE" \
  --latent-channels "$LATENT_CHANNELS" \
  --n-spectrum-points "$N_SPECTRUM_POINTS" \
  --autoencoder-checkpoint-dir "$AE_RUN_VAE" \
  --use-vae \
  --run-name "$RUN_NAME" \
  --manual-seed "$MANUAL_SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"
DIFF_RUN_VAE="$(latest_run_dir results_smoke/latent_diffusion_vae)"

# 6) cDCGAN baseline
run_cmd "$PYTHON_BIN" src/baseline/train_cdcgan.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results_smoke/cdcgan \
  --epochs "$CDCGAN_E" \
  --batch-size "$BATCH_SIZE" \
  --n-spectrum-points "$N_SPECTRUM_POINTS" \
  --run-name "$RUN_NAME" \
  --manual-seed "$MANUAL_SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"
CDCGAN_RUN="$(latest_run_dir results_smoke/cdcgan)"

# 7) Evaluation checks
run_cmd "$PYTHON_BIN" src/evaluation/run_experiments.py \
  --device "$DEVICE" \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --run-dir "$FP_RUN" \
  --output-json "$FP_RUN/eval_forward.json" \
  forward \
  --max-batches 2

run_cmd "$PYTHON_BIN" src/evaluation/run_experiments.py \
  --device "$DEVICE" \
  --target-spectra-csv "$SPECTRA_CSV" \
  --run-dir "$DIFF_RUN" \
  --forward-predictor-ckpt "$FP_RUN/forward_predictor.pth" \
  --output-json "$DIFF_RUN/eval_one_to_many_smoke.json" \
  one-to-many \
  --use-ema \
  --n-targets 2 \
  --n-samples 2 \
  --n-steps 50 \
  --guidance-scale 0.5

run_cmd "$PYTHON_BIN" src/evaluation/run_experiments.py \
  --device "$DEVICE" \
  --target-spectra-csv "$SPECTRA_CSV" \
  --run-dir "$DIFF_RUN_VAE" \
  --forward-predictor-ckpt "$FP_RUN/forward_predictor.pth" \
  --output-json "$DIFF_RUN_VAE/eval_one_to_many_smoke.json" \
  one-to-many \
  --use-ema \
  --n-targets 2 \
  --n-samples 2 \
  --n-steps 50 \
  --guidance-scale 0.5

## OOD smoke evaluation disabled by default
# if [[ -n "$OOD_IMG_DIR" && -n "$OOD_SPECTRA_CSV" ]]; then
#   echo "Running OOD smoke evaluation with OOD_IMG_DIR=$OOD_IMG_DIR and OOD_SPECTRA_CSV=$OOD_SPECTRA_CSV"
#   run_cmd "$PYTHON_BIN" src/evaluation/run_experiments.py \
#     --device "$DEVICE" \
#     --img-dir "$OOD_IMG_DIR" \
#     --spectra-csv "$OOD_SPECTRA_CSV" \
#     --run-dir "$FP_RUN" \
#     --output-json "$FP_RUN/eval_ood_smoke.json" \
#     ood \
#     --tolerance 0.1
# else
#   echo "Skipping OOD smoke evaluation (set OOD_IMG_DIR and OOD_SPECTRA_CSV to enable)."
# fi

run_cmd "$PYTHON_BIN" src/evaluation/demo_inference.py \
  --run-dir "$DIFF_RUN" \
  --spectra-csv "$SPECTRA_CSV" \
  --n-targets 2 \
  --n-samples 2 \
  --n-steps 50 \
  --device "$DEVICE" \
  --use-ema

echo
echo "============================================================"
echo "PIPELINE CHECK COMPLETE"
echo "Forward run:       $FP_RUN"
echo "AE run:            $AE_RUN"
echo "AE VAE run:        $AE_RUN_VAE"
echo "Diffusion run:     $DIFF_RUN"
echo "Diffusion VAE run: $DIFF_RUN_VAE"
echo "cDCGAN run:        $CDCGAN_RUN"
echo "Master log:        $LOG_FILE"
echo "============================================================"
