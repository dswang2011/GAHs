#!/bin/bash
#SBATCH --job-name=ROTTENTOMATOES
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=80G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=2-00

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
source activate python36
cd /home/vbd667/code/GAHs
python train.py --dataset ROTTENTOMATOES --max_sequence_length 50
