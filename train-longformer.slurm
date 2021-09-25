#!/bin/bash
#SBATCH --job-name=longformer-news-polarity
#SBATCH --partition=GPU
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:V100S:1

# Load python distribution and CUDA
module load anaconda3
module load cuda/11.2

# Activate local venv
conda activate pytorch
echo $CONDA_DEFAULT_ENV

# Bash command for python script
python -m train.py
