#!/bin/bash
#SBATCH --job-name=yelp20_1
#SBATCH --ntasks=1 --cpus-per-task=5 --mem=100G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=2-00

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
source activate python36
cd /home/vbd667/code/GAHs

# python train.py --dataset SUBJ --max_sequence_length 50 --cus_pos n --k_roles 5 --search_times 5 
# python train.py --dataset SUBJ --max_sequence_length 50 --cus_pos n --k_roles 4 --search_times 25
# python train.py --dataset SUBJ --max_sequence_length 50 --cus_pos n --k_roles 3 --search_times 10
# python train.py --dataset SUBJ --max_sequence_length 50 --cus_pos n --k_roles 2 --search_times 10
# python train.py --dataset SUBJ --max_sequence_length 50 --cus_pos n --k_roles 1 --search_times 8 --tag_encoding 1

# python train.py --dataset AGNews --max_sequence_length 50 --cus_pos n --k_roles 5
# python train.py --dataset AGNews --max_sequence_length 50 --cus_pos n --k_roles 4
# python train.py --dataset AGNews --max_sequence_length 50 --cus_pos n --k_roles 3
# python train.py --dataset AGNews --max_sequence_length 50 --cus_pos n --k_roles 2
# python train.py --dataset AGNews --max_sequence_length 50 --cus_pos n --k_roles 1

# python train_large.py --dataset TREC --max_sequence_length 60 --cus_pos n --k_roles 5 --search_times 12
# python train.py --dataset TREC --max_sequence_length 60 --cus_pos n --k_roles 4 --search_times 25
# python train.py --dataset TREC --max_sequence_length 60 --cus_pos n --k_roles 3
# python train.py --dataset TREC --max_sequence_length 60 --cus_pos n --k_roles 2
# python train.py --dataset TREC --max_sequence_length 60 --cus_pos n --k_roles 1 --search_times 10 --tag_encoding 0

# python train.py --dataset IMDB --max_sequence_length 270 --cus_pos a --k_roles 5 --search_times 6
# python train.py --dataset IMDB --max_sequence_length 270 --cus_pos a --k_roles 4 --search_times 30
# python train.py --dataset IMDB --max_sequence_length 270 --cus_pos a --k_roles 3
# python train.py --dataset IMDB --max_sequence_length 270 --cus_pos a --k_roles 2
# python train.py --dataset IMDB --max_sequence_length 270 --cus_pos a --k_roles 1 --search_times 7 --tag_encoding 1

# python train.py --dataset SST --max_sequence_length 50 --cus_pos a --k_roles 1 --search_times 11 --tag_encoding 1
# python train.py --dataset SST --max_sequence_length 50 --cus_pos a --k_roles 4 --search_times 25 --tag_encoding 0

# python train.py --dataset MR --max_sequence_length 50 --cus_pos a --k_roles 5 --search_times 15
# python train.py --dataset MR --max_sequence_length 50 --cus_pos a --k_roles 4 --search_times 25
# python train.py --dataset MR --max_sequence_length 50 --cus_pos a --k_roles 3 --search_times 10
# python train.py --dataset MR --max_sequence_length 50 --cus_pos a --k_roles 2 --search_times 10
# python train.py --dataset MR --max_sequence_length 50 --cus_pos a --k_roles 1 --search_times 10 --tag_encoding 1
# python train.py --dataset ROTTENTOMATOES --max_sequence_length 90 --cus_pos a --k_roles 5 --search_times 15
# python train.py --dataset ROTTENTOMATOES --max_sequence_length 90 --cus_pos a --k_roles 4 --search_times 25
# python train.py --dataset ROTTENTOMATOES --max_sequence_length 90 --cus_pos a --k_roles 3
# python train.py --dataset ROTTENTOMATOES --max_sequence_length 90 --cus_pos a --k_roles 2
# python train.py --dataset ROTTENTOMATOES --max_sequence_length 90 --cus_pos a --k_roles 1 --search_times 10 --tag_encoding 1

# python train.py --dataset DBPEDIA --max_sequence_length 50 --cus_pos n --k_roles 5 --search_times 5
# python train.py --dataset DBPEDIA --max_sequence_length 50 --cus_pos n --k_roles 4 --search_times 25
# python train.py --dataset DBPEDIA --max_sequence_length 50 --cus_pos n --k_roles 3
# python train.py --dataset DBPEDIA --max_sequence_length 50 --cus_pos n --k_roles 2
# python train.py --dataset DBPEDIA --max_sequence_length 50 --cus_pos n --k_roles 1 --search_times 10 --tag_encoding 1

python train.py --dataset YELP --max_sequence_length 150 --cus_pos a --k_roles 5 --search_times 6 --tag_encoding 0
python train.py --dataset YELP --max_sequence_length 150 --cus_pos a --k_roles 4 --search_times 28
# python train.py --dataset YELP --max_sequence_length 150 --cus_pos a --k_roles 3
# python train.py --dataset YELP --max_sequence_length 150 --cus_pos a --k_roles 1 --search_times 15 --tag_encoding 0