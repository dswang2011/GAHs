"""
This is role oriented mask generation

"""

class RoleMask(self):
	def __init__(self, opt):
		self.opt = opt

		# POS tag category
		self.noun_list = ['NN','NNS','NNP','NNPS']
		self.verb_list = ['VB', 'VBZ', 'VBD', 'VBG','VBN','VBP']
		self.adjective_list = ['JJ','JJR','JJS']
		self.punctuations = [';','?',',','.',':']


	def getMask(self,role='positional'):
		
