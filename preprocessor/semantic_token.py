"""
This is to preprocess the raw files into semantic annotated token list.
by Dongsheng, April 6, 2020
"""

import raw_data_loader
import spacy
import pickle  


nlp = spacy.load("en")


class SemToken(self):
	def __init__(self,dataset):
		self.dataset = opt.dataset
		self.train = train_file_path
		self.test = test_file_path
		self.ispair = self.opt.pair_set.split(",")

		self.prep_train_path = 'prepared/'+dataset+'/train.pkl'
		self.prep_test_path = 'prepared/'+dataset+'/test.pkl'

	# text to lit of semtoks
	def get_semtok_list(self,text):
		doc = nlp(text.strip())
		return [token for token in doc]

	# texts to semtoks
	def semtok_on_texts(self,texts):
		return [self.get_semtok_list(text) for text in texts]

	# process the data
	def process_file(self):
		# open
		train_texts,train_labels = raw_data_loader.load_data(self.dataset,'train')
		test_texts,test_labels = raw_data_loader.load_data(self.dataset,'test')

		# process and save
		if self.ispair:
			train_texts1,train_texts2 = self.semtok_on_texts(train_texts[0]),self.semtok_on_texts(train_texts[1])
			test_texts1,test_texts2 = self.semtok_on_texts(test_texts[0]),self.semtok_on_texts(test_texts[1])
			# save
			pickle.dump([[train_texts1,train_texts2],train_labels],open(self.prep_train_path, 'wb'))
			pickle.dump([[test_texts1,test_texts2],test_labels],open(self.prep_test_path,'wb'))
		else:
			train_texts = self.semtok_on_texts(train_texts)
			test_texts = self.semtok_on_texts(test_texts)
			# save
			pickle.dump([train_texts,train_labels],open(self.prep_train_path, 'wb'))
			pickle.dump([test_texts,test_labels],open(self.prep_test_path,'wb'))


	if __name__ == '__main__':
		# fb data samples
		for dataset in  ['']:
			process_file(file)