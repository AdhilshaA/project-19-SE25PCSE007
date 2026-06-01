#!/usr/bin/env bash
set -euo pipefail

# Full research pipeline for training, evaluation, and testing.
# Run from project root:
#   bash scripts/all_experiments.sh
#
# Optional environment overrides (examples):
#   DEVICE=cuda BATCH_SIZE=16 bash scripts/all_experiments.sh
#   PYTHON_BIN=python3 bash scripts/all_experiments.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
IMG_DIR="${IMG_DIR:-data/Images}"
SPECTRA_CSV="${SPECTRA_CSV:-data/absorptionData_HybridGAN.csv}"
MANUAL_SEED="${MANUAL_SEED:-999}"
RUN_NAME="${RUN_NAME:-full}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
# Optional dedicated OOD dataset (leave empty to skip OOD step)
OOD_IMG_DIR="${OOD_IMG_DIR:-}"
OOD_SPECTRA_CSV="${OOD_SPECTRA_CSV:-}"

# Shared model/training defaults
BATCH_SIZE="${BATCH_SIZE:-16}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
LATENT_SIZE="${LATENT_SIZE:-16}"
LATENT_CHANNELS="${LATENT_CHANNELS:-1}"
N_SPECTRUM_POINTS="${N_SPECTRUM_POINTS:-800}"

# Full-run epochs
FP_EPOCHS="${FP_EPOCHS:-25}"
AE_EPOCHS="${AE_EPOCHS:-40}"
DIFF_EPOCHS="${DIFF_EPOCHS:-20}"
CDCGAN_EPOCHS="${CDCGAN_EPOCHS:-100}"

# Evaluation defaults
N_TARGETS="${N_TARGETS:-20}"
N_SAMPLES="${N_SAMPLES:-8}"
N_STEPS_EVAL="${N_STEPS_EVAL:-1000}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"

LOG_DIR="results/experiment_logs"
mkdir -p "$LOG_DIR"
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/all_experiments_${RUN_TAG}.log"

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

run_cmd() {
  echo
  echo "[CMD] $*"
  "$@"
}

echo "============================================================"
echo "FULL EXPERIMENT PIPELINE"
echo "Log file: $LOG_FILE"
echo "============================================================"

check_required_paths

# 1) Forward predictor training
run_cmd "$PYTHON_BIN" src/training/train_forward_predictor.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results/forward_predictor \
  --epochs "$FP_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --n-spectrum-points "$N_SPECTRUM_POINTS" \
  --run-name "$RUN_NAME" \
  --manual-seed "$MANUAL_SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"

FP_RUN="$(latest_run_dir results/forward_predictor)"
echo "Forward predictor run: $FP_RUN"

# 2) Deterministic autoencoder training
run_cmd "$PYTHON_BIN" src/training/train_design_autoencoder.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results/design_autoencoder_det \
  --epochs "$AE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --latent-size "$LATENT_SIZE" \
  --latent-channels "$LATENT_CHANNELS" \
  --run-name "$RUN_NAME" \
  --manual-seed "$MANUAL_SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"

AE_RUN_DET="$(latest_run_dir results/design_autoencoder_det)"
echo "Deterministic AE run: $AE_RUN_DET"

# 3) Conv-VAE autoencoder ablation
run_cmd "$PYTHON_BIN" src/training/train_design_autoencoder.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results/design_autoencoder_vae \
  --epochs "$AE_EPOCHS" \
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
  --kl-warmup-epochs 5

AE_RUN_VAE="$(latest_run_dir results/design_autoencoder_vae)"
echo "Conv-VAE run: $AE_RUN_VAE"

# 4) Diffusion (deterministic AE latents)
run_cmd "$PYTHON_BIN" src/training/train_diffusion.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results/latent_diffusion_det \
  --epochs "$DIFF_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --latent-size "$LATENT_SIZE" \
  --latent-channels "$LATENT_CHANNELS" \
  --n-spectrum-points "$N_SPECTRUM_POINTS" \
  --autoencoder-checkpoint-dir "$AE_RUN_DET" \
  --run-name "$RUN_NAME" \
  --manual-seed "$MANUAL_SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"

DIFF_RUN_DET="$(latest_run_dir results/latent_diffusion_det)"
echo "Diffusion (det) run: $DIFF_RUN_DET"

# 5) Diffusion (Conv-VAE latents)
run_cmd "$PYTHON_BIN" src/training/train_diffusion.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results/latent_diffusion_vae \
  --epochs "$DIFF_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --latent-size "$LATENT_SIZE" \
  --latent-channels "$LATENT_CHANNELS" \
  --n-spectrum-points "$N_SPECTRUM_POINTS" \
  --autoencoder-checkpoint-dir "$AE_RUN_VAE" \
  --manual-seed "$MANUAL_SEED" \
  --use-vae \
  --run-name "$RUN_NAME" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"

DIFF_RUN_VAE="$(latest_run_dir results/latent_diffusion_vae)"
echo "Diffusion (VAE) run: $DIFF_RUN_VAE"

# 7) Evaluate forward predictor
run_cmd "$PYTHON_BIN" src/evaluation/run_experiments.py \
  --device "$DEVICE" \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --run-dir "$FP_RUN" \
  --output-json "$FP_RUN/eval_forward.json" \
  forward

# 8) One-to-many eval for deterministic diffusion (EMA + guided)
run_cmd "$PYTHON_BIN" src/evaluation/run_experiments.py \
  --device "$DEVICE" \
  --target-spectra-csv "$SPECTRA_CSV" \
  --run-dir "$DIFF_RUN_DET" \
  --forward-predictor-ckpt "$FP_RUN/forward_predictor.pth" \
  --output-json "$DIFF_RUN_DET/eval_one_to_many_ema_guided.json" \
  one-to-many \
  --use-ema \
  --n-targets "$N_TARGETS" \
  --n-samples "$N_SAMPLES" \
  --n-steps "$N_STEPS_EVAL" \
  --guidance-scale "$GUIDANCE_SCALE"

# 9) One-to-many eval ablation for deterministic diffusion (EMA + no guidance)
run_cmd "$PYTHON_BIN" src/evaluation/run_experiments.py \
  --device "$DEVICE" \
  --target-spectra-csv "$SPECTRA_CSV" \
  --run-dir "$DIFF_RUN_DET" \
  --forward-predictor-ckpt "$FP_RUN/forward_predictor.pth" \
  --output-json "$DIFF_RUN_DET/eval_one_to_many_ema_noguide.json" \
  one-to-many \
  --use-ema \
  --n-targets "$N_TARGETS" \
  --n-samples "$N_SAMPLES" \
  --n-steps "$N_STEPS_EVAL" \
  --guidance-scale 0.0

# 10) OOD eval for deterministic diffusion (EMA) - disabled
# if [[ -n "$OOD_IMG_DIR" && -n "$OOD_SPECTRA_CSV" ]]; then
#   echo "Running OOD evaluation with OOD_IMG_DIR=$OOD_IMG_DIR and OOD_SPECTRA_CSV=$OOD_SPECTRA_CSV"
#   run_cmd "$PYTHON_BIN" src/evaluation/run_experiments.py \
#     --device "$DEVICE" \
#     --img-dir "$OOD_IMG_DIR" \
#     --spectra-csv "$OOD_SPECTRA_CSV" \
#     --run-dir "$FP_RUN" \
#     --output-json "$FP_RUN/eval_ood.json" \
#     ood \
#     --tolerance 0.1
# else
#   echo "Skipping OOD evaluation (set OOD_IMG_DIR and OOD_SPECTRA_CSV to enable)."
# fi

# 11) One-to-many eval for VAE diffusion (EMA + guided)
run_cmd "$PYTHON_BIN" src/evaluation/run_experiments.py \
  --device "$DEVICE" \
  --target-spectra-csv "$SPECTRA_CSV" \
  --run-dir "$DIFF_RUN_VAE" \
  --forward-predictor-ckpt "$FP_RUN/forward_predictor.pth" \
  --output-json "$DIFF_RUN_VAE/eval_one_to_many_ema_guided.json" \
  one-to-many \
  --use-ema \
  --n-targets "$N_TARGETS" \
  --n-samples "$N_SAMPLES" \
  --n-steps "$N_STEPS_EVAL" \
  --guidance-scale "$GUIDANCE_SCALE"

# 12) Demo inference with figures and metrics for deterministic diffusion
run_cmd "$PYTHON_BIN" src/evaluation/demo_inference.py \
  --run-dir "$DIFF_RUN_DET" \
  --spectra-csv "$SPECTRA_CSV" \
  --n-targets 5 \
  --n-samples 6 \
  --n-steps "$N_STEPS_EVAL" \
  --device "$DEVICE" \
  --use-ema

# 6) cDCGAN baseline
run_cmd "$PYTHON_BIN" src/baseline/train_cdcgan.py \
  --img-dir "$IMG_DIR" \
  --spectra-csv "$SPECTRA_CSV" \
  --output-dir results/cdcgan \
  --epochs "$CDCGAN_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --n-spectrum-points "$N_SPECTRUM_POINTS" \
  --run-name "$RUN_NAME" \
  --manual-seed "$MANUAL_SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"

CDCGAN_RUN="$(latest_run_dir results/cdcgan)"
echo "cDCGAN run: $CDCGAN_RUN"

echo
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Forward run:        $FP_RUN"
echo "AE det run:         $AE_RUN_DET"
echo "AE vae run:         $AE_RUN_VAE"
echo "Diffusion det run:  $DIFF_RUN_DET"
echo "Diffusion vae run:  $DIFF_RUN_VAE"
echo "cDCGAN run:         $CDCGAN_RUN"
echo "Master log:         $LOG_FILE"
echo "============================================================"
