# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from tqdm import tqdm
from keras.layers import Conv1D, MaxPooling1D,Dense,  LSTM, GRU, Bidirectional,Dropout,Input,GlobalMaxPooling1D, Embedding,Concatenate
from models.BasicModel import BasicModel
from keras.models import Model
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
from keras import backend as K
import random

add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])

# checked
class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape

# It's safe to use a 1-d mask for self-attention; checked
class ScaledDotProductAttention():
	def __init__(self, d_model, attn_dropout=0.1):
		self.temper = np.sqrt(d_model)
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k, v, mask):   # mask_k or mask_qk
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k]) # shape=(batch, q, k)
		if mask is not None:
			mmask = Lambda(lambda x:(-1e+10)*(1.-K.cast(x, 'float32')))(mask)
			# mmask = Lambda(lambda x:(-1e+10)*(1-x)) (mask)
			attn = add_layer([attn, mmask])
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output, attn

# checked
class MultiHeadAttention():
	# mode 0 - big martixes, faster; mode 1 - more clear implementation
	def __init__(self, n_head, d_model, dropout, use_norm=False):
		self.n_head = n_head
		self.d_k = self.d_v = d_k = d_v = d_model // n_head
		self.dropout = dropout

		# attention mode 1
		self.qs_layers = []
		self.ks_layers = []
		self.vs_layers = []
		for _ in range(n_head):
			self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
			self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
			self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))

		self.attention = ScaledDotProductAttention(d_model)
		self.layer_norm = LayerNormalization() if use_norm else None
		self.w_o = TimeDistributed(Dense(d_model))

	def __call__(self, q, k, v, masks=None):
		d_k, d_v = self.d_k, self.d_v
		n_head = self.n_head

		# heads
		heads = []; attns = []
		for i in range(n_head):
			qs = self.qs_layers[i](q)   
			ks = self.ks_layers[i](k) 
			vs = self.vs_layers[i](v)
			if masks!=None and i<len(masks): 
				head,attn = self.attention(qs, ks, vs, masks[i])
			else:
				head, attn = self.attention(qs, ks, vs, None)
			heads.append(head); attns.append(attn)
		head = Concatenate()(heads) if n_head > 1 else heads[0]
		attn = Concatenate()(attns) if n_head > 1 else attns[0]

		outputs = self.w_o(head)
		outputs = Dropout(self.dropout)(outputs)
		return outputs, attn


# checked
class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = LayerNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)

# one block = (multi-head self-attention + normalization) +  feedforward
class BlockEncoder():
	def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
		self.norm_layer = LayerNormalization()
	def __call__(self, enc_input, masks=None):	# enc_input = src_embed
		# multi self-attention+norm
		output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, masks=masks)
		output = self.norm_layer(Add()([enc_input, output]))
		# feedforward
		output = self.pos_ffn_layer(output)
		return output, slf_attn

class PosEncodingLayer:
	def __init__(self, max_len, d_emb):
		self.pos_emb_matrix = Embedding(max_len, d_emb, trainable=False, weights=[GetPosEncodingMatrix(max_len, d_emb)])
	def get_pos_seq(self, x):
		mask = K.cast(K.not_equal(x, 0), 'int32')
		pos = K.cumsum(K.ones_like(x, 'int32'), 1)
		return pos * mask
	def __call__(self, seq, pos_input=False):
		x = seq
		if not pos_input: x = Lambda(self.get_pos_seq)(x)
		return self.pos_emb_matrix(x)

# util functions
def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc

def GetPadMask(q, k):
	'''
	shape: [B, Q, K]
	'''
	ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
	mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
	mask = K.batch_dot(ones, mask, axes=[2,1])
	return mask

def GetSubMask(s):
	'''
	shape: [B, Q, K], lower triangle because the i-th row should have i 1s.
	'''
	len_s = tf.shape(s)[1]
	bs = tf.shape(s)[:1]
	mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
	return mask

# checked
class MultiLayerEncoder():
	def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
		self.n_head = n_head
		self.layers = [BlockEncoder(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
	def __call__(self, src_emb, src_seq, return_att=False, active_layers=999, masks = None):
		if return_att: atts = []
		pad_mask = Lambda(lambda x:GetPadMask(x, x))(src_seq)
		all_masks = []
		for mask in masks: all_masks.append(mask)
		for i in range(self.n_head-len(masks)): all_masks.append(pad_mask)

		x = src_emb		
		for i,enc_layer in enumerate(self.layers[:active_layers]):
			# if i>1:
			# 	x, att = enc_layer(x, None)
			# else:	
			x, att = enc_layer(x, all_masks)

			if return_att: atts.append(att)
		return (x, atts) if return_att else x


class GAHs(BasicModel):
	def get_model(self,opt, active_layers=999) :   
		# prepared to be used
		self.src_seq = Input(shape=(opt.max_sequence_length,), dtype='int32')
		self.masks = [Input(shape=(opt.max_sequence_length,opt.max_sequence_length),dtype='float32') for i in range(len(opt.roles))]

		self.pos_emb = PosEncodingLayer(opt.max_sequence_length, opt.embedding_dim)# if self.src_loc_info else None
		self.emb_dropout = Dropout(opt.dropout_rate)
		self.word_emb = Embedding(len(opt.word_index) + 1,opt.embedding_dim,weights=[opt.embedding_matrix],input_length=opt.max_sequence_length,trainable=True)
		self.multi_layer_encoder = MultiLayerEncoder(opt.embedding_dim, opt.d_inner_hid, opt.n_head, opt.layers, opt.dropout_rate)
		self.meaner=Lambda(lambda x: K.mean(x, axis=-2) )
		self.predict = Dense(opt.nb_classes, activation='softmax')

		# ensembling part
		src_emb = self.word_emb(self.src_seq)
		if True: 
			src_emb = add([src_emb, self.pos_emb(self.src_seq)])
		
		src_emb = self.emb_dropout(src_emb)

		# NGA (global)
		# randomly choose some mask combinations
		# ...
		# sample a combination
		self.mask_comb = []
		self.sample_i = random.sample(range(len(self.masks)),k=3)
		for i in self.sample_i:
			self.mask_comb.append(self.masks[i])
		# encoder
		enc_output = self.multi_layer_encoder(src_emb, self.src_seq, active_layers=active_layers, masks = self.mask_comb)
		# enc_output = Dense(200,activation='relu')(enc_output)
		# dence and predict
		mean_output= self.meaner(enc_output)
		preds = self.predict(mean_output)   # 3 catetory
		return Model([self.src_seq]+self.masks, preds)
