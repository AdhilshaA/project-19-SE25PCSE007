#!/usr/bin/env bash
set -euo pipefail

# Full orchestration script with ablations and figure generation.
# Usage: ./scripts/run_full_evaluation.sh <RUN_NAME> [SEED]

RUN_NAME=${1:-full_report}
SEED=${2:-999}
PYTHON=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda}
IMG_DIR=${IMG_DIR:-data/Images}
SPECTRA_CSV=${SPECTRA_CSV:-data/absorptionData_HybridGAN.csv}

# Training defaults (full)
FP_EPOCHS=${FP_EPOCHS:-25}
AE_EPOCHS=${AE_EPOCHS:-40}
DIFF_EPOCHS=${DIFF_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-16}
HIDDEN_DIM=${HIDDEN_DIM:-64}
LATENT_SIZE=${LATENT_SIZE:-16}
LATENT_CHANNELS=${LATENT_CHANNELS:-1}
N_SPECTRUM_POINTS=${N_SPECTRUM_POINTS:-800}

# Evaluation + figure defaults
N_STEPS_EVAL=${N_STEPS_EVAL:-1000}
N_TARGETS_MAIN=${N_TARGETS_MAIN:-20}
N_SAMPLES_MAIN=${N_SAMPLES_MAIN:-8}
N_TARGETS_DEMO=${N_TARGETS_DEMO:-1}
GUIDANCE_SCALES=${GUIDANCE_SCALES:-"0.0 1.0"}
GUIDANCE_START_FRACTION=${GUIDANCE_START_FRACTION:-0.3}
GUIDANCE_NUM_APPLICATIONS=${GUIDANCE_NUM_APPLICATIONS:-7}
RUN_CDCGAN=${RUN_CDCGAN:-0}
CDCGAN_EPOCHS=${CDCGAN_EPOCHS:-100}

OUT_BASE="results/report_${RUN_NAME}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_BASE"

latest_run() {
  local base="$1"
  if [[ -d "$base" ]]; then
    ls -dt "$base"/run_* 2>/dev/null | head -n1 || true
  fi
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp -v "$src" "$dst"
  fi
}

copy_latest_demo_dir() {
  local target_dir="$1"
  local label="$2"
  local latest_demo
  latest_demo=$(ls -dt results/demo_* 2>/dev/null | head -n1 || true)
  if [[ -n "$latest_demo" && -d "$latest_demo" ]]; then
    cp -rv "$latest_demo" "$target_dir/${label}_$(basename "$latest_demo")"
  fi
}

run_one_to_many_eval() {
  local run_dir="$1"
  local out_json="$2"
  local use_ema="$3"
  local guidance_scale="$4"
  local n_targets="$5"
  local n_samples="$6"

  local cmd=(
    "$PYTHON" src/evaluation/run_experiments.py
    --device "$DEVICE"
    --target-spectra-csv "$SPECTRA_CSV"
    --run-dir "$run_dir"
    --forward-predictor-ckpt "$FP_RUN/forward_predictor.pth"
    --output-json "$out_json"
    one-to-many
    --n-targets "$n_targets"
    --n-samples "$n_samples"
    --n-steps "$N_STEPS_EVAL"
    --guidance-scale "$guidance_scale"
    --guidance-start-fraction "$GUIDANCE_START_FRACTION"
    --guidance-num-applications "$GUIDANCE_NUM_APPLICATIONS"
  )
  if [[ "$use_ema" == "1" ]]; then
    cmd+=(--use-ema)
  fi
  "${cmd[@]}"
}

echo "Report base: $OUT_BASE"

if [[ ! -d "$IMG_DIR" ]]; then
  echo "Error: image directory not found: $IMG_DIR"
  exit 1
fi
if [[ ! -f "$SPECTRA_CSV" ]]; then
  echo "Error: spectra CSV not found: $SPECTRA_CSV"
  exit 1
fi

echo "==> Train forward predictor"
"$PYTHON" src/training/train_forward_predictor.py \
  --img-dir "$IMG_DIR" --spectra-csv "$SPECTRA_CSV" \
  --output-dir results/forward_predictor --epochs "$FP_EPOCHS" --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" --n-spectrum-points "$N_SPECTRUM_POINTS" \
  --run-name "${RUN_NAME}_fp" --manual-seed "$SEED"
FP_RUN=$(latest_run results/forward_predictor)
mkdir -p "$OUT_BASE/forward_predictor"
copy_if_exists "$FP_RUN/epoch_metrics.csv" "$OUT_BASE/forward_predictor/"
copy_if_exists "$FP_RUN/loss_curve.png" "$OUT_BASE/forward_predictor/"
copy_if_exists "$FP_RUN/metric_curve.png" "$OUT_BASE/forward_predictor/"
copy_if_exists "$FP_RUN/test_metrics.json" "$OUT_BASE/forward_predictor/"

echo "==> Train deterministic Autoencoder (AE)"
"$PYTHON" src/training/train_design_autoencoder.py \
  --img-dir "$IMG_DIR" --spectra-csv "$SPECTRA_CSV" \
  --output-dir results/design_autoencoder_det --epochs "$AE_EPOCHS" --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" --latent-size "$LATENT_SIZE" --latent-channels "$LATENT_CHANNELS" \
  --run-name "${RUN_NAME}_ae_det" --manual-seed "$SEED"
AE_DET_RUN=$(latest_run results/design_autoencoder_det)
mkdir -p "$OUT_BASE/design_autoencoder_det"
copy_if_exists "$AE_DET_RUN/epoch_metrics.csv" "$OUT_BASE/design_autoencoder_det/"
copy_if_exists "$AE_DET_RUN/loss_curve.png" "$OUT_BASE/design_autoencoder_det/"
copy_if_exists "$AE_DET_RUN/test_metrics.json" "$OUT_BASE/design_autoencoder_det/"

echo "==> Train VAE Autoencoder (VAE)"
"$PYTHON" src/training/train_design_autoencoder.py \
  --img-dir "$IMG_DIR" --spectra-csv "$SPECTRA_CSV" \
  --output-dir results/design_autoencoder_vae --epochs "$AE_EPOCHS" --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" --latent-size "$LATENT_SIZE" --latent-channels "$LATENT_CHANNELS" \
  --vae --beta 0.05 --kl-warmup-epochs 20 --kl-free-bits 0.03 --kl-normalize-by-dim \
  --run-name "${RUN_NAME}_ae_vae" --manual-seed "$SEED"
AE_VAE_RUN=$(latest_run results/design_autoencoder_vae)
mkdir -p "$OUT_BASE/design_autoencoder_vae"
copy_if_exists "$AE_VAE_RUN/epoch_metrics.csv" "$OUT_BASE/design_autoencoder_vae/"
copy_if_exists "$AE_VAE_RUN/loss_curve.png" "$OUT_BASE/design_autoencoder_vae/"
copy_if_exists "$AE_VAE_RUN/test_metrics.json" "$OUT_BASE/design_autoencoder_vae/"

echo "==> Train diffusion ablation matrix (VAE/non-VAE x EMA on/off)"
for AE_KIND in det vae; do
  if [[ "$AE_KIND" == "det" ]]; then
    AE_RUN="$AE_DET_RUN"
  else
    AE_RUN="$AE_VAE_RUN"
  fi

  for EMA_TRAIN in on off; do
    DIFF_OUT="results/latent_diffusion_${AE_KIND}_${EMA_TRAIN}"
    DIFF_TAG="diff_${AE_KIND}_${EMA_TRAIN}"
    DIFF_RUN_NAME="${RUN_NAME}_${DIFF_TAG}"

    DIFF_FLAGS=(
      --img-dir "$IMG_DIR" --spectra-csv "$SPECTRA_CSV"
      --output-dir "$DIFF_OUT" --epochs "$DIFF_EPOCHS" --batch-size "$BATCH_SIZE"
      --hidden-dim "$HIDDEN_DIM" --latent-size "$LATENT_SIZE" --latent-channels "$LATENT_CHANNELS"
      --n-spectrum-points "$N_SPECTRUM_POINTS"
      --autoencoder-checkpoint-dir "$AE_RUN"
      --run-name "$DIFF_RUN_NAME" --manual-seed "$SEED"
    )
    if [[ "$AE_KIND" == "vae" ]]; then
      DIFF_FLAGS+=(--use-vae)
    fi
    if [[ "$EMA_TRAIN" == "off" ]]; then
      DIFF_FLAGS+=(--disable-ema)
    fi

    echo "==> Train diffusion: $DIFF_TAG"
    "$PYTHON" src/training/train_diffusion.py "${DIFF_FLAGS[@]}"
    DIFF_RUN=$(latest_run "$DIFF_OUT")

    RUN_ART_DIR="$OUT_BASE/${DIFF_TAG}"
    mkdir -p "$RUN_ART_DIR/eval"
    copy_if_exists "$DIFF_RUN/epoch_metrics.csv" "$RUN_ART_DIR/"
    copy_if_exists "$DIFF_RUN/loss_curve.png" "$RUN_ART_DIR/"
    copy_if_exists "$DIFF_RUN/test_metrics.json" "$RUN_ART_DIR/"
    copy_if_exists "$DIFF_RUN/args.json" "$RUN_ART_DIR/"

    echo "==> One-to-many evaluations: EMA/no-EMA x guidance scales"
    for SAMPLE_EMA in 0 1; do
      EMA_TAG="noema"
      if [[ "$SAMPLE_EMA" == "1" ]]; then EMA_TAG="ema"; fi
      for GS in $GUIDANCE_SCALES; do
        OUT_JSON="$RUN_ART_DIR/eval/one_to_many_${EMA_TAG}_guide_${GS}.json"
        run_one_to_many_eval "$DIFF_RUN" "$OUT_JSON" "$SAMPLE_EMA" "$GS" "$N_TARGETS_MAIN" "$N_SAMPLES_MAIN" || true
      done
      # Explicit one-to-many with 5 designs per spectrum.
      OUT_JSON_5="$RUN_ART_DIR/eval/one_to_many_${EMA_TAG}_5samples.json"
      run_one_to_many_eval "$DIFF_RUN" "$OUT_JSON_5" "$SAMPLE_EMA" "1.0" "$N_TARGETS_MAIN" "5" || true
    done

    echo "==> Figure generation (single, 5-sample, 16-sample)"
    for DEMO_SAMPLES in 1 5 16; do
      DEMO_TAG="samples_${DEMO_SAMPLES}"
      "$PYTHON" src/evaluation/demo_inference.py \
        --run-dir "$DIFF_RUN" --spectra-csv "$SPECTRA_CSV" \
        --n-targets "$N_TARGETS_DEMO" --n-samples "$DEMO_SAMPLES" \
        --n-steps "$N_STEPS_EVAL" --device "$DEVICE" || true
      copy_latest_demo_dir "$RUN_ART_DIR" "${DEMO_TAG}_noema"

      "$PYTHON" src/evaluation/demo_inference.py \
        --run-dir "$DIFF_RUN" --spectra-csv "$SPECTRA_CSV" \
        --n-targets "$N_TARGETS_DEMO" --n-samples "$DEMO_SAMPLES" \
        --n-steps "$N_STEPS_EVAL" --device "$DEVICE" --use-ema || true
      copy_latest_demo_dir "$RUN_ART_DIR" "${DEMO_TAG}_ema"
    done
  done
done

if [[ "$RUN_CDCGAN" == "1" ]]; then
  echo "==> Train cDCGAN baseline"
  "$PYTHON" src/baseline/train_cdcgan.py \
    --img-dir "$IMG_DIR" --spectra-csv "$SPECTRA_CSV" \
    --output-dir results/cdcgan --epochs "$CDCGAN_EPOCHS" --batch-size "$BATCH_SIZE" \
    --n-spectrum-points "$N_SPECTRUM_POINTS" \
    --run-name "${RUN_NAME}_cdcgan" --manual-seed "$SEED"
  CDCGAN_RUN=$(latest_run results/cdcgan)
  mkdir -p "$OUT_BASE/cdcgan"
  copy_if_exists "$CDCGAN_RUN/epoch_metrics.csv" "$OUT_BASE/cdcgan/"
  copy_if_exists "$CDCGAN_RUN/generator_loss_curve.png" "$OUT_BASE/cdcgan/"
  copy_if_exists "$CDCGAN_RUN/discriminator_loss_curve.png" "$OUT_BASE/cdcgan/"
fi

echo "Artifacts collected in $OUT_BASE"
echo "Run complete."
