"""
This is role oriented mask generation

"""
import numpy as np

class RoleMask(object):
	def __init__(self, opt):
		self.opt = opt

		# POS tag category
		self.noun_list = ['NN','NNS','NNP','NNPS']
		self.verb_list = ['VB', 'VBZ', 'VBD', 'VBG','VBN','VBP']
		self.adjective_list = ['JJ','JJR','JJS']

		self.punctuations = [';','?',',','.',':']
		self.major_rels = ['nsubj', 'dobj', 'amod', 'advmod']


	def enable_neibor(self,mask,i,neib_num,MAX_SEQUENCE_LENGTH):
		for j in range(neib_num):
			if i+j<MAX_SEQUENCE_LENGTH: mask[i][i+j]=1.
			if i-j>=0: mask[i][i-j]=1.

	def positional_masks_of_texts(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH,neib_num=2):
		masks = [np.zeros((MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH)) for i in range(len(tokens_lists))]
		for text_id, text in enumerate(tokens_lists):
			mask = masks[text_id]
			# START
			for i in range(min(len(text)+2,MAX_SEQUENCE_LENGTH)):
				self.enable_neibor(mask,i,neib_num,MAX_SEQUENCE_LENGTH)
		return np.asarray(masks,dtype='float32')


	def POS_masks_of_texts(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = [np.zeros((MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH)) for i in range(len(tokens_lists))]
		include_tags = self.noun_list #+ self.verb_list + self.adjective_list
		for tid, text in enumerate(tokens_lists):
			mask = masks[tid]
			# Start
			val_index = [0]
			# Body
			i=1
			for semtok in text:
				if semtok.tag_ in include_tags:
					val_index.append(i)
				i+=1
			# END
			if i<MAX_SEQUENCE_LENGTH:
				val_index.append(i)
			# assign val part
			for m in val_index:
				for n in val_index:
					mask[m][n]=1.
		return np.asarray(masks,dtype='float32')

	def major_rel_of_texts(self,tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = [np.zeros((MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH)) for i in range(len(tokens_lists))]
		include_tags = self.major_rels
		for tid, text in enumerate(tokens_lists):
			mask = masks[tid]
			val_index = [0]	
			i = 1
			for semtok in text:
				if semtok.dep_ in include_tags: val_index.append(i)
				i+=1
			# END
			if i<MAX_SEQUENCE_LENGTH:
				val_index.append(i)
			# assign
			for m in val_index:
				for n in val_index:
					mask[m][n]=1.
		return np.asarray(masks,dtype='float32')


	def both_direct_masks_of_texts(self,tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
		masks = [np.zeros((MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH)) for i in range(len(tokens_lists))]
		for text_id, text in enumerate(tokens_lists):
			# for each text or sentence
			mask = masks[text_id]
			mask[0][0]=1.
			i = 1
			for semtok in text:
				looks = [i]
				looks.append(semtok.head.i+1)	# parent
				looks+=[child.i+1 for child in semtok.children]	# children
				looks+=[sib.i+1 for sib in semtok.head.children]	# siblings
				looks = list(set(looks))
				for look in looks:
					mask[i][look]=1.
				i+=1
			if i<MAX_SEQUENCE_LENGTH: mask[i][i]=1.
		return np.asarray(masks,dtype='float32')

	# separator and punctuations
	def separator_mask(self,tokens_lists,word_index,MAX_SEQUENCE_LENGTH):
		masks = [np.zeros((MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH)) for i in range(len(tokens_lists))]
		for text_id, text in enumerate(tokens_lists):
			# for each text or sentence
			mask = masks[text_id]
			# get all separators and punctuations
			sep = [0]
			i=1
			for semtok in text:
				if semtok.text in self.punctuations: sep.append(i)
				i+=1
			if i<MAX_SEQUENCE_LENGTH: sep.append(i)
			# assign
			for m in range(min(len(text)+2,MAX_SEQUENCE_LENGTH)):
				for n in sep:
					mask[m][n]=1.
		return np.asarray(masks,dtype='float32')


	def get_masks(self,tokens_lists,word_index, MAX_SEQUENCE_LENGTH,mask_list=['major_rels']):
		res = []
		for mask in mask_list:
			if mask == 'major_rels':
				res.append(self.major_rel_of_texts(tokens_lists,word_index, MAX_SEQUENCE_LENGTH))
			elif mask == 'positional':
				res.append(self.positional_masks_of_texts(tokens_lists,word_index, MAX_SEQUENCE_LENGTH))
			elif mask == 'POS':
				res.append(self.POS_masks_of_texts(tokens_lists,word_index, MAX_SEQUENCE_LENGTH))
			elif mask == 'both_direct':
				res.append(self.both_direct_masks_of_texts(tokens_lists,word_index, MAX_SEQUENCE_LENGTH))
			elif mask == 'separator':
				res.append(self.separator_mask(tokens_lists,word_index, MAX_SEQUENCE_LENGTH))
		return res

	# broad attend
	# def positional_masks_of_texts(self, tokens_lists, word_index, MAX_SEQUENCE_LENGTH):
	# 	masks = [np.ones((MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH)) for i in range(len(tokens_lists))]
	# 	for tid, tokens in enumerate(tokens_lists):
	# 		mask = masks[tid]
	# 		# START 
	# 		max_seq = max(MAX_SEQUENCE_LENGTH,len(toknes))
	# 		mask[:max_seq] = 0.0

	# 	return np.asarray(masks,dtype='float32')