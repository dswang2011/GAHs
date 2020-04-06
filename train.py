"""
This is to train the model
by Dongsheng, April 6, 2020
"""

from  Params import Params
import argparse
import models
import numpy as np

def train_single(opt):
	model = models.setup(opt)

def train_grid(opt):
	# grid search
	for i in range(args.search_times):
        print('search num: {}'.format(i))
        for key in grid_pool:
            value = random.choice(grid_pool[key])
            setattr(config,key, value)
            merged_dict[key] = value
            print(key,':',value)
        model = models.setup(opt)



if __name__ == '__main__':
    # initialize paras
    params = Params()
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=6)
	args = parser.parse_args()
	params.parse_config(args.config)

    train_single(params)
