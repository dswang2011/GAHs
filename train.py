"""
This is to train the model
by Dongsheng, April 6, 2020
"""

from  Params import Params
import argparse
import models
import numpy as np
import data_helper
import util
import random

dataset_pool = {
	"TREC": ['train','test'],
	"MR": ['train']	
}
grid_pool ={
	# model
	"model": ["cnn",'bilstm','Transformer'],
	"hidden_unit_num":[50,100,150,200],	# for rnn or cnn
	"dropout_rate" : [0.2,0.3,0.4,0.5],
	# hyper parameters
	"lr":[0.01, 0.001, 0.0008],
	"batch_size":[32,64,96],
	"val_split":[0.15],
	"layers" : [2,4,6,8],
	"n_head" : [1,2,4,6],
	"d_inner_hid" : [128,256,512]
}

def train_single(opt):
	# choose a random combinations
	para_str = ''
	for key in grid_pool:
		value = random.choice(grid_pool[key])
		setattr(opt,key, value)
		para_str+=key
		para_str+= str(value)+'_'
	setattr(opt,'para_str',para_str)
	print('[parameters]:',para_str)
	# load input
	data = data_helper.Data_helper(opt)
	dataset = 'TREC'
	train_test = data.load_data(dataset, dataset_pool[dataset])
	# set model and train
	model = models.setup(opt)
	if len(dataset_pool[dataset])>1:
		train,test = train_test
	else:
		[train],test = train_test, None 
	model.train(train,dev=test,dataset = dataset)


def train_grid(opt):
	# load input
	data = data_helper.Data_helper(opt)
	dataset = 'MR'
	train_test = data.load_data(dataset,dataset_pool[dataset])

	# search N times:
	memo = set()
	for i in range(args.search_times):
		print('[search time]:', i,'/',args.search_times)
		para_str = ''
		for key in grid_pool:
			value = random.choice(grid_pool[key])
			setattr(opt,key, value)
			para_str+=key
			para_str+= str(value)+'_'
		# memo skip
		if para_str in memo: continue
		memo.add(para_str)
		# else run this paras
		print('[parameters]:',para_str)
		setattr(opt,'para_str',para_str)
		# set model and train
		model = models.setup(opt)
		if len(dataset_pool[dataset])>1:
			train,test = train_test
		else:
			print('lenth of train_Test:',len(train_test))
			[train],test = train_test, None 
		model.train(train,dev=test,dataset = dataset)


if __name__ == '__main__':
	# initialize paras
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=6)
	parser.add_argument('--epoch_num', type=int, default=10)
	parser.add_argument('--search_times', type=int, default=15)
	args = parser.parse_args()
	# set parameters from config files
	util.parse_and_set(args.config,args)
	# train
	train_grid(args)
