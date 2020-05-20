"""
This is to train the model
by Dongsheng, April 6, 2020
"""

import argparse
import models
import numpy as np
import data_larger
import util
import random
import tensorflow as tf

dataset_pool = {
	"TREC": ['train','test'],	# 50
	"MR": ['train','test'],	# 50	
	"SST": ['train','test'],	# 19
	"IMDB":['train','test'],	# 230
	"YELP": ['train','test'],	# 136
	"ROTTENTOMATOES": ['train','test'], # 21
	"DBPEDIA" : ['train','test'],	# 47
	"AGNews": ['train','test'],
	"SUBJ":['train','test'],	# avg length
	# AG news 8, 
}
grid_pool ={
	# model
	"model": ['gahs'],#,'gahs','gahs','gah'],#,],#,'gah' ,'cnn','bilstm',
	"hidden_unit_num":[100,200,300],	# for rnn or cnn
	"dropout_rate" : [0.2,0.3,0.4],
	# hyper parameters
	"lr":[0.001, 0.0005,0.0001],	# 0.01 for CNN and LSTM
	"batch_size":[32,64,96],
	"val_split":[0.1],
	"layers" : [2,4,6,8],
	"n_head" : [6,8],
	"d_inner_hid" : [128,256,512],
	"roles": [['positional','both_direct','major_rels','separator','rare_word']]	#,,,'POS' ,rare_word,'positional','stop_word',,'both_direct','major_rels','separator','rare_word'
}

# 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		# Restrict TensorFlow to only use the fourth GPU
		tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
	# Memory growth must be set before GPUs have been initialized
		print(e)


def train_single(opt):
	# choose a random combinations
	para_str = ''
	for key in grid_pool:
		value = random.choice(grid_pool[key])
		setattr(opt, key, value)
		para_str+= key
		para_str+= str(value)+'_'
	para_str = opt.dataset+'_'+para_str
	print('[paras]:',para_str)
	setattr(opt,'para_str',para_str)
	# load input
	if 'gah' in opt.model: opt.load_role = True
	data = data_helper.Data_helper(opt)
	train_test = data.load_train(opt.dataset, dataset_pool[opt.dataset])
	# set model and train
	model = models.setup(opt)
	if len(dataset_pool[opt.dataset])>1:
		train,test = train_test
	else:
		[train],test = train_test, None 
	model.train(train,dev=test,dataset = opt.dataset)


def train_grid(opt):
	# search N times:
	memo = set()
	for i in range(args.search_times):
		print('[search time]:', i,'/',args.search_times)
		para_str = ''
		for key in grid_pool:
			value = random.choice(grid_pool[key])
			setattr(opt, key, value)
			para_str+= key
			para_str+= str(value)+'_'
		# memo skip
		# if para_str in memo: continue
		memo.add(para_str)

		# load input
		data = data_larger.Data_helper(opt)
		train_gene, test_gene = data.load_train(opt.dataset,dataset_pool[opt.dataset])

		# else run this paras
		print('[paras]:',para_str)
		setattr(opt,'para_str',para_str)
		# set model and train
		model = models.setup(opt)
		# model.train_large(train_gene,test_gene, dataset = opt.dataset)

		# sem encoding
		if opt.tag_encoding==1:
			opt.para_str += 'semEmbed'
			model.train_tag(train_gene,dev=test_gene,dataset = opt.dataset)
		else:
			model.train_large(train_gene,dev=test_gene,dataset = opt.dataset)


if __name__ == '__main__':
	# initialize paras
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=5)
	parser.add_argument('--epoch_num', type=int, default=30)
	parser.add_argument('--search_times', type=int, default=20)
	parser.add_argument('--load_role',type=bool, default=False)
	parser.add_argument('--dataset', default="MR")
	parser.add_argument('--max_sequence_length', type=int,default=90)
	parser.add_argument('--k_roles', type=int,default=5)
	parser.add_argument('--cus_pos',default='N')
	parser.add_argument('--tag_encoding',type=int,default=0)
	args = parser.parse_args()
	# set parameters from config files
	util.parse_and_set(args.config,args)
	# train
	print('== Currently train set is:==', args.dataset)
	print('=== tag encoding:', args.tag_encoding)
	
	train_grid(args)
