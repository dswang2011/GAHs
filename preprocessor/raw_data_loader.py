import os
import numpy as np


class RawLoader(self):
	__init__(self,opt):
		self.opt = opt

	def load_IMDB_data(self,file_type='train'):
		texts = []
		labels = []
		# read data

		return texts,labels



	def load_WNLI_data(self,file_type='train'):
		texts1,texts2=[],[]
		labels=[]
		if file_type == 'train':
			file = self.opt.wnli_train_path
		else:
			file = self.opt.wnli_test_path

		with open(file, 'r', encoding='utf8', errors='ignore') as f:
			for row in f:
		
		return [texts1,texts2],labels


	# the only one
	def load_data(self,dataset,file_type="train"):
		if dataset in ['IMDB']:
			texts,labels = load_IMDB_date()
		elif dataset in ['WNLI']:
			texts,labels = load_WNLI_data(file_path = output_root+file_name)
		# other files

		return texts,labels
