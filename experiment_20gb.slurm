#!/bin/bash
#SBATCH --job-name=hybridgan_smoke
#SBATCH --output=logs/hybridgan_%j.out
#SBATCH --error=logs/hybridgan_%j.err

#SBATCH --partition=gpu_scholar
#SBATCH --gres=gpu:a100_3g.20gb:1

#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

echo "====================================="
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "DATE: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "====================================="

nvidia-smi -L
nvidia-smi

cd $SLURM_SUBMIT_DIR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genai_gpu

echo "Python: $(which python)"
echo "Torch check:"
python -c "
import torch
print('Torch:', torch.__version__)
print('CUDA Build:', torch.version.cuda)
print('CUDA Available:', torch.cuda.is_available())
"

bash scripts/all_experiments.sh
