"""
This file is to load the data for training
by Dongsheng, 2020, Aprial 07
"""

import os
#import stanfordnlp
import numpy as np
import codecs

import pickle  
from  Params import Params
import argparse
from keras.utils import to_categorical
import numpy as np
from sklearn import preprocessing
import util
from mask import RoleMask
import re

punctuation_list = [',',':',';','.','!','?','...','…','。']


class Data_helper(object):
	def __init__(self,opt):
		self.opt=opt  
		self.role_mask = RoleMask(self.opt)
		self.is_numberic = re.compile(r'^[-+]?[0-9.]+$')

	def get_embedding_dict(self, GLOVE_DIR):
		embeddings_index = {}
		f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf-8")
		for line in f:
			if line.strip()=='':
				continue
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()
		# customized dict
		f =  codecs.open(os.path.join(GLOVE_DIR, 'customized.100d.txt'),encoding="utf-8")  #
		for line in f:
			if line.strip()=='':
				continue
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
		f.close()
		return embeddings_index

	def build_word_embedding_matrix(self,word_index):
		# word embedding lodading
		embeddings_index = self.get_embedding_dict(self.opt.glove_dir)
		print('Total %s word vectors.' % len(embeddings_index))

		# initial: random initial (not zero initial)
		embedding_matrix = np.random.random((len(word_index) + 1,self.opt.embedding_dim ))
		for word, i in word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				embedding_matrix[i] = embedding_vector
		return embedding_matrix

	def load_sem_data(self,dataset,split):
		root = 'datasets/'+dataset+'/'
		texts,labels = pickle.load(open(os.path.join(root,split+'.pkl'),'rb'))
		return texts,labels

	# load the train, valid or test
	def load_data(self,dataset,splits):
		texts_list_train_test = []
		labels_train_test = []
		for split in splits:
			texts,labels = self.load_sem_data(dataset,split)
			texts_list_train_test.append(texts)
			labels_train_test.append(labels)
		self.opt.nb_classes = len(set(labels))
		print('[LABEL]',self.opt.nb_classes, ' labels:',set(labels))
		# max_num_words = self.opt.max_num_words
		if dataset in self.opt.pair_set.split(","):
			all_texts= [set(sentence) for texts1,texts2 in texts_list_train_test for sentence in texts1]
		else:
			all_texts= [set(sentence) for dataset in texts_list_train_test for sentence in dataset]
		# tokenize 
		word_index = self.tokenizer(all_texts,MAX_NB_WORDS=self.opt.max_nb_words)
		# save word_index
		self.opt.word_index = word_index
		print('word_index:',len(word_index))
		# build word embedding
		self.opt.embedding_matrix = self.build_word_embedding_matrix(word_index)

		le = preprocessing.LabelEncoder()
		# labels = le.fit_transform(labels)
		# padding
		train_test = []
		for tokens_list,labels in zip(texts_list_train_test,labels_train_test):
			if dataset in self.opt.pair_set.split(","):
				x1 = self.tokens_list_to_sequences(tokens_list[0],word_index,self.opt.max_sequence_length)
				x2 = self.tokens_list_to_sequences(tokens_list[1],word_index,self.opt.max_sequence_length)
				x = [x1,x2]
			else:
				x = self.tokens_list_to_sequences(tokens_list,word_index,self.opt.max_sequence_length)
				# if load_role, then load masks as well
				if self.opt.load_role:
					masks = self.role_mask.get_masks(tokens_list,word_index,self.opt.max_sequence_length, self.opt.all_roles)
					x = [x]+masks
			y = le.fit_transform(labels)
			# print(y)
			y = to_categorical(np.asarray(y)) # one-hot encoding y_train = labels # one-hot label encoding
			train_test.append([x,y])
			if dataset in self.opt.pair_set.split(","):
				print('[train pair] Shape of data tensor:', x[0].shape,' and ', x[1].shape)
			else:
				print('[train] Shape of data tensor:', x[0].shape)
			print('[train] Shape of label tensor:', y.shape)

		return train_test

	def tokenizer(self, texts, MAX_NB_WORDS):
		word_index = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<MASK>':3, '<NUM>':4}
		index = 5
		for text in texts:
			for token in text:	# here the text is the doc
				# add to word_index
				if len(word_index)<MAX_NB_WORDS:
					token=token.text.lower()
					if token not in word_index.keys():
						word_index[token] = index
						index+=1
		return word_index

	def clean_str(self, string):
	    """
	    Tokenization/string cleaning for dataset
	    Every dataset is lower cased except
	    """
	    string = re.sub(r"\\", "", string)    
	    string = re.sub(r"\'", "", string)    
	    string = re.sub(r"\"", "", string)    
	    return string.strip().lower()


	# input is the generalized text; 
	def tokens_list_to_sequences(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		sequences = []
		for tokens in tokens_lists:
			sequence = [1]
			for semtok in tokens:
				token = semtok.text.lower()
				if token in word_index.keys():
					token_index = word_index[token]
					sequence.append(token_index)
				# elif self.is_numberic.match(value):
				# 	sequence.append(4)
			sequence.append(2)
			if len(sequence)>MAX_SEQUENCE_LENGTH:
				sequence = sequence[:MAX_SEQUENCE_LENGTH]
			else:
				sequence = sequence+np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()
			# print('seq:',sequence)
			sequences.append(sequence)
		return np.asarray(sequences,dtype=int)



if __name__ == '__main__':
		# initialize paras
	parser = argparse.ArgumentParser(description='run the training.')
	parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
	parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
	parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
	parser.add_argument('--patience', type=int, default=6)
	args = parser.parse_args()
	# set parameters from config files
	util.parse_and_set(args.config,args)

	data_help = Data_helper(args)
	train,test = data_help.load_data('TREC')

