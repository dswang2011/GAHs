"""
This is to train the model
by Dongsheng, April 6, 2020
"""

from  Params import Params
import argparse
import models
import numpy as np
import data_helper

def train_single(opt):
	grid_pool ={
		# model
		"model": ["transformer"],

		"hidden_unit_num":[50,100,200],	# for rnn or cnn
		"dropout_rate" : [0.2,0.3,0.4,0.5],

		# hyper parameters
		"lr":[0.01,0.001],
		"batch_size":[32,64,96],
		"validation_split":[0.1,0.15,0.2],
		"layers" : [4,6,8],
		"n_head" : [4,6,8,12],
		"d_inner_hid" : [128,256,512]
	}
	# choose a random combinations
	for key in grid_pool:
		value = random.choice(grid_pool[key])
		setattr(opt,key, value)
		print(key,':',value)

	# load input
	data = Data_helper(opt)
	dataset = 'WNLI'
	train,test = datasets.load_data(dataset)
	# set model and train
	model = models.setup(opt)
	model.train(train,dev=test,dataset = dataset)


# def train_grid(opt):
# 	# grid search
# 	for i in range(args.search_times):
# 		print('search num: {}'.format(i))
# 		for key in grid_pool:
# 			value = random.choice(grid_pool[key])
# 			setattr(config,key, value)
# 			merged_dict[key] = value
# 			print(key,':',value)
# 		model = models.setup(opt)


if __name__ == '__main__':
	# initialize paras
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=6)
	args = parser.parse_args()

	train_single(args)
