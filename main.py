
import argparse
import models
import numpy as np
import data_helper
import util
import random
from preprocessor.semantic_token import SemToken
import train



if __name__ == '__main__':
	# initialize paras
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=6)
	parser.add_argument('--epoch_num', type=int, default=10)
	parser.add_argument('--search_times', type=int, default=20)
	parser.add_argument('--load_role',type=bool, default=False)
	parser.add_argument('--run_mode',default='preprocess')
	parser.add_argument('--dataset', default="TREC")
	parser.add_argument('--splits', default="train,test")
	args = parser.parse_args()
	# set parameters from config files
	util.parse_and_set(args.config,args)
	# train
	if args.run_mode == 'preprocess':
		semTok = SemToken(args)
		# custom
		dataset = args.dataset
		splits = args.splits.split(',')
		semTok.process_file(dataset, splits)

	if args.run_mode == "train":
		train.train_grid(args)


