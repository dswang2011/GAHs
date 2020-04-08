"""
Read raw data, you can specify your data file path here or in config.init

"""


import os
import numpy as np
import tensorflow_datasets as tfds


class RawLoader(self):
	__init__(self,opt):
		self.opt = opt

	def load_IMDB_data(self,file_path, split='train'):
		texts = []
		labels = []
		
		# read data
		# ds = tfds.load('imdb_reviews', split='train')
		# for ex in ds.take(4):
  # 			print(ex)
		return texts,labels


	def load_WNLI_data(self,file_path, split='train'):
		texts1,texts2=[],[]
		labels=[]
		if split == 'train':
			file = self.opt.wnli_train_path
		elif split =='valid':
			file = self.opt.wnli_valid_path
		elif split=='test':
			ile = self.opt.wnli_test_path

		with open(file, 'r', encoding='utf8', errors='ignore') as f:
			for row in f:
		
		return [texts1,texts2],labels


	# the only one
	def load_data(self,dataset,split="train"):
		root = 'datasets/'
		if dataset in ['IMDB']:
			texts,labels = self.load_IMDB_date()
		elif dataset in ['WNLI']:
			texts,labels = self.load_WNLI_data()
		# other files

		return texts,labels