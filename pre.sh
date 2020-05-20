#!/bin/bash
#SBATCH --job-name=pre-YELP
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=110G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=2-00

vbd667
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
source activate python36
cd /home/vbd667/code/GAHs
# python main.py --run_mode preprocess --dataset MR --splits train
python main.py --run_mode prepare_feed --dataset YELP --splits train,test --max_sequence_length 150
# python main.py --run_mode prepare_feed --dataset IMDB --splits train,test --max_sequence_length 270
# python main.py --run_mode prepare_feed --dataset TREC --splits train,test --max_sequence_length 60
