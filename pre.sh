#!/bin/bash
#SBATCH --job-name=yelp
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=30G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=2-00

vbd667
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
source activate python36
cd /home/vbd667/code/GAHs
python main.py --run_mode preprocess --dataset YELP --splits test
