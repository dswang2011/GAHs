"""
Read raw data, you can specify your data file path here or in config.init
"""


import os
import numpy as np
# import tensorflow_datasets as tfds


class RawLoader(object):
	def __init__(self,opt):
		self.opt = opt


	def load_WNLI_data(self,file_path, split='train'):
		texts1,texts2=[],[]
		labels=[]
		if split == 'train':
			file = self.opt.wnli_train_path
		elif split =='valid':
			file = self.opt.wnli_valid_path
		elif split=='test':
			ile = self.opt.wnli_test_path

		# with open(file, 'r', encoding='utf8', errors='ignore') as f:
		# 	for row in f:
		
		return [texts1,texts2],labels

	def load_TREC_data(self, split='train'):
		texts,labels = [],[]
		if split=='train':
			file_path = 'datasets/TREC/TREC.train.all'
		elif split=='test':
			file_path = 'datasets/TREC/TREC.test.all'
		with open(file_path,'r',encoding='utf8') as fr:
			for line in fr:
				line = self.processed_text(line)
				strs = line.strip().split(' ',1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts,labels


	def load_MR_data(self,split='train'):
		texts,labels = [],[]
		file_path = 'datasets/MR/rt-polarity.all'
		with open(file_path,'r',encoding='utf8') as fr:
			for line in fr:
				line = self.processed_text(line)
				strs = line.strip().split(' ',1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts,labels

	def load_IMDB_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/IMDB/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts, labels

	def load_YELP_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/YELP/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts, labels

	def load_DBPEDIA_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/DBPEDIA/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts, labels

	def load_SST_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/SST/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts, labels

	def load_ROTTENTOMATOES_data(self, split='train'):
		texts, labels = [], []
		folder_path = 'datasets/ROTTENTOMATOES/'
		with open(folder_path+split+'.csv', 'r', encoding='utf8') as fr:
			for line in fr:
				# line = self.processed_text(line)
				strs = line.strip().split(',', 1)
				texts.append(strs[1])
				labels.append(strs[0])
		return texts, labels
	# the only one
	def load_data(self,dataset,split="train"):
		root = 'datasets/'
		if dataset in ['IMDB']:
			texts,labels = self.load_IMDB_data()
		elif dataset in ['YELP']:
			texts,labels = self.load_YELP_data()
		elif dataset in ['SST']:
			texts,labels = self.load_SST_data()
		elif dataset in ['ROTTENTOMATOES']:
			texts,labels = self.load_ROTTENTOMATOES_data()
		elif dataset in ['DBPEDIA']:
			texts,labels = self.load_DBPEDIA_data()
		elif dataset in ['WNLI']:
			texts,labels = self.load_WNLI_data()
		# other files
		elif dataset == 'MR':
			texts,labels = self.load_MR_data()
		elif dataset == 'TREC':
			texts,labels = self.load_TREC_data(split=split)
		return texts,labels

	def processed_text(self,text):
		text = text.replace('\\\\', '')
		text = text.replace('\n','')
		#stripped = strip_accents(text.lower())
		text = text.lower()
		return text

if __name__ == '__main__':
	rawLoader = RawLoader(None)
	texts,labels = rawLoader.load_data('TREC',split='train')
	print(len(texts),len(labels))
	print(set(labels))
	print(texts[:5])