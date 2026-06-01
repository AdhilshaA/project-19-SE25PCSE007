#!/usr/bin/env bash
set -euo pipefail

# Ablation sweep: train AE det/VAE, train diffusion with/without EMA,
# run evaluations with guidance scales and collect artifacts.
# Usage: ./scripts/run_ablation_sweep.sh <RUN_NAME> [SEED]

RUN_NAME=${1:-ablation}
SEED=${2:-999}
PYTHON=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda}
OUT_BASE="results/ablation_${RUN_NAME}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_BASE"

latest_run() {
  local base="$1"
  if [ -d "$base" ]; then
    ls -dt "$base"/run_* 2>/dev/null | head -n1 || true
  fi
}

echo "Ablation out: $OUT_BASE"

# Train forward predictor once
echo "==> Train forward predictor"
$PYTHON src/training/train_forward_predictor.py \
  --img-dir data/Images --spectra-csv data/absorptionData_HybridGAN.csv \
  --output-dir results/forward_predictor --epochs 25 --batch-size 16 \
  --run-name "${RUN_NAME}_fp" --manual-seed "$SEED" --device "$DEVICE"
FP_RUN=$(latest_run results/forward_predictor)
mkdir -p "$OUT_BASE/forward_predictor"
cp -v "$FP_RUN"/epoch_metrics.csv "$OUT_BASE/forward_predictor/" 2>/dev/null || true
cp -v "$FP_RUN"/loss_curve.png "$OUT_BASE/forward_predictor/" 2>/dev/null || true

# Train both AE variants
for VAE_FLAG in "" "--vae"; do
  if [ "$VAE_FLAG" = "" ]; then
    AE_OUT=results/design_autoencoder_det
    AE_TAG="ae_det"
  else
    AE_OUT=results/design_autoencoder_vae
    AE_TAG="ae_vae"
  fi

  echo "==> Train AE ($AE_TAG)"
  $PYTHON src/training/train_design_autoencoder.py \
    --img-dir data/Images --spectra-csv data/absorptionData_HybridGAN.csv \
    --output-dir $AE_OUT --epochs 40 --batch-size 16 \
    --hidden-dim 64 --latent-size 16 --latent-channels 1 \
    $VAE_FLAG --beta 0.05 --kl-warmup-epochs 20 --kl-free-bits 0.03 --kl-normalize-by-dim \
    --run-name "${RUN_NAME}_${AE_TAG}" --manual-seed "$SEED" --device "$DEVICE"

  AE_RUN=$(latest_run $AE_OUT)
  mkdir -p "$OUT_BASE/$AE_TAG"
  cp -v "$AE_RUN"/epoch_metrics.csv "$OUT_BASE/$AE_TAG/" 2>/dev/null || true
  cp -v "$AE_RUN"/loss_curve.png "$OUT_BASE/$AE_TAG/" 2>/dev/null || true

  # For each AE type, train diffusion with and without EMA
  for EMA in "on" "off"; do
    EMA_FLAG=""
    if [ "$EMA" = "off" ]; then
      EMA_FLAG="--disable-ema"
    fi
    DIFF_OUT=results/latent_diffusion_${AE_TAG}_${EMA}
    DIFF_TAG="diff_${AE_TAG}_${EMA}"

    echo "==> Train diffusion ($DIFF_TAG) using AE run $AE_RUN (EMA=$EMA)"
    $PYTHON src/training/train_diffusion.py \
      --img-dir data/Images --spectra-csv data/absorptionData_HybridGAN.csv \
      --output-dir $DIFF_OUT --epochs 20 --batch-size 16 \
      --hidden-dim 64 --latent-size 16 --latent-channels 1 --n-spectrum-points 800 \
      --autoencoder-checkpoint-dir "$AE_RUN" \
      --run-name "${RUN_NAME}_${DIFF_TAG}" --manual-seed "$SEED" --device "$DEVICE" $EMA_FLAG

    DIFF_RUN=$(latest_run $DIFF_OUT)
    mkdir -p "$OUT_BASE/$DIFF_TAG"
    cp -v "$DIFF_RUN"/epoch_metrics.csv "$OUT_BASE/$DIFF_TAG/" 2>/dev/null || true
    cp -v "$DIFF_RUN"/loss_curve.png "$OUT_BASE/$DIFF_TAG/" 2>/dev/null || true

    # Evaluate with guidance on/off
    for GS in 0.0 1.0; do
      outjson="$OUT_BASE/eval_${DIFF_TAG}_guide_${GS}.json"
      echo "Eval: $DIFF_RUN | EMA=$EMA | guidance=$GS -> $outjson"
      $PYTHON src/evaluation/run_experiments.py \
        --device "$DEVICE" --target-spectra-csv data/absorptionData_HybridGAN.csv \
        --run-dir "$DIFF_RUN" --forward-predictor-ckpt "$FP_RUN/forward_predictor.pth" \
        --use-ema \
        --n-targets 20 --n-samples 8 --n-steps 1000 --guidance-scale $GS \
        --output-json "$outjson" one-to-many || true
    done

    # Generate demo images (combined) for the diffusion run
    echo "Generating demo_inference outputs for $DIFF_RUN"
    $PYTHON src/evaluation/demo_inference.py \
      --run-dir "$DIFF_RUN" --spectra-csv data/absorptionData_HybridGAN.csv \
      --n-targets 5 --n-samples 16 --n-steps 1000 --device "$DEVICE" --use-ema || true
    cp -v results/demo_* "$OUT_BASE/" 2>/dev/null || true
  done

done

echo "Ablation artifacts collected in $OUT_BASE"
