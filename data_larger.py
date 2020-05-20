"""
This file is to load the data for training
by Dongsheng, 2020, Aprial 07
"""

import os
#import stanfordnlp
import numpy as np
import codecs

import pickle
import argparse
from keras.utils import to_categorical
import numpy as np
from sklearn import preprocessing
import util
from mask import RoleMask
import re
import gc
from preprocessor.DataGenerator import DataGenerator

punctuation_list = [',',':',';','.','!','?','...','…','。']


class Data_helper(object):
	def __init__(self,opt):
		self.opt=opt
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

	def build_tag_embedding_matrix(self,tag_onehot):
		tag_embedding_matrix = np.random.random((len(tag_onehot)+1,self.opt.dep_dim))
		for i,vect in enumerate(tag_onehot):
			if vect is not None:
				tag_embedding_matrix[i] = np.array(vect)
		return tag_embedding_matrix


	def load_sem_data(self,dataset,split):
		root = 'datasets/'+dataset+'/'
		texts,labels = pickle.load(open(os.path.join(root,split+'.pkl'),'rb'))
		return texts,labels


	# load the train, valid or test
	def load_train(self,dataset,splits):
		# common paras
		partition, labels, word_index,tag_index,dep_dim, nb_classes = pickle.load(open(os.path.join('datasets/',dataset,'comm.pkl'),'rb'))
		self.opt.word_index = word_index
		self.opt.nb_classes = nb_classes
		
		print('word_index:',len(word_index))
		# build word embedding
		self.opt.embedding_matrix = self.build_word_embedding_matrix(word_index)

		# tag embedding
		tag_onehot = to_categorical( list(tag_index.values()) )
		self.opt.dep_dim = len(tag_onehot[0])
		self.opt.dep_embedding_matrix = self.build_tag_embedding_matrix(tag_onehot)
		

		# Generators
		training_generator = DataGenerator(self.opt,'train', partition['train'], labels)
		test_generator = DataGenerator(self.opt,'test', partition['test'], labels)

		return [training_generator,test_generator]


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

	def tag_index(self, texts, MAX_NB_WORDS):
		tag_index = {'<PAD>': 0}
		index = 1
		count = 0
		for text in texts:
			for token in text:	# here the text is the doc
				# add to word_index
				if len(tag_index)<100:	# less than 100
					tag=token.dep_
					if tag not in tag_index.keys():
						tag_index[tag] = index
						index+=1
				else:
					break
			count+=1
			if count>2000: break
		return tag_index


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
			sequence = [1]	# start
			for semtok in tokens:
				token = semtok.text.lower()
				if self.is_numberic.match(token):
					sequence.append(4)
				elif token in word_index.keys():
					token_index = word_index[token]
					sequence.append(token_index)
				else:
					sequence.append(0)
				
			sequence.append(2)	# end
			if len(sequence)>MAX_SEQUENCE_LENGTH:
				sequence = sequence[:MAX_SEQUENCE_LENGTH]
			else:
				sequence = sequence+np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()
			# print('seq:',sequence)
			sequences.append(sequence)
		return np.asarray(sequences,dtype=int)
		# return sequences

	def tokens_list_to_tag_sequences(self, tokens_lists, tag_index, MAX_SEQUENCE_LENGTH):
		sequences = []
		for tokens in tokens_lists:
			sequence = [0]	# start
			for semtok in tokens:
				tag = semtok.dep_
				if tag in tag_index.keys():
					index = tag_index[tag]
					sequence.append(index)
				else:
					sequence.append(0)
			sequence.append(0)	# end
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
	parser.add_argument('--load_role',type=bool, default=True)
	parser.add_argument('--all_roles', default=['positional','both_direct','major_rels','stop_word'])
	
	args = parser.parse_args()
	# set parameters from config files
	util.parse_and_set(args.config,args)

	data_help = Data_helper(args)
	splits = ['train','test']
	train,test = data_help.load_data('TREC', splits)

