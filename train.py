import os
from typing import Tuple, List, Dict
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from collections import Counter
from tensorflow.keras.layers import Dense, Input,Masking,LSTM, Embedding,Reshape, Dropout, Activation,TimeDistributed,Bidirectional,concatenate, GlobalMaxPool1D
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
import pickle
from tensorflow.keras.optimizers import SGD
from  tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import collections
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
unigram_path = 'msr_unigram.utf8'
X_train_path = 'input_msr_training'
Y_train_path = 'label_msr_training'

unigram_vocab = dict()
unigram_word_to_id = dict()
X_train_uni = []
Y_train = []
Batch_size = 256
num_class = 4
lstm_dim=256
def vocabulary(unigram_path=unigram_path ):
	"""
	This is the function to build the vocabulary of the dataset.

	:param unigram_path: The path to the file that contains the unigrams
	:return: None
	"""
	with open(unigram_path, 'r', encoding='utf8') as f:
	  original_lines = f.readlines()
	  for line in original_lines:
	  	words = line.split()
	  	for word in words:
	  		if word not in unigram_vocab:
	  			unigram_vocab[word] = 1
	  		else:
	  			unigram_vocab[word] += 1

def word2index():
	"""
	Converts each character to its index in the vocabulary

	:return: None
	"""
	vocabulary()
	unigram_word_to_id["<PAD>"] = 0 #zero is not casual!
	unigram_word_to_id["<UNK>"] = 1 #OOV are mapped as <UNK>
	unigram_word_to_id.update({k:v+len(unigram_word_to_id) for k, v in unigram_vocab.items()})

def tokenize_dataset(X_train_path=X_train_path):
	"""
	Converts each character to its index in the vocabulary

	:param X_train_path: path to the trainig set with no spaces
	:return: encoded X  training set
	"""
	word2index()
	with open(X_train_path, 'r', encoding='utf8') as f:
	  original_lines = f.readlines()
	  original_lines = [line.replace("\u3000","")for line in original_lines]
	  for line in original_lines:
	  	words = line.split()
	  	for word in words:
	  		char = []
	  		for c in word:
	  			try:
	  				char.append(unigram_word_to_id[c])
	  			except KeyError:
	  				char.append(unigram_word_to_id["<UNK>"])
	  	X_train_uni.append(char)
	return X_train_uni

def convert_labels_to_integer(string):
	"""
	Converts the labels from BIES format to integer
	:param string: integer to be converted to BIES format
	:return: Array of labels in Integer
	"""
	tags = []
	for word in string.strip():
	    if word == 'S':
	        tags.append(3)  # 'S', a Single character
	    elif word == 'B':
	        tags.append(0)  # 'B', Begin of a word
	    elif word == 'I':
	        tags.append(1)  # 'I', Middle of a word
	    elif word == 'E':
	        tags.append(2)  # 'E',  End of a word
	    else:
	        continue
	return tags
def my_to_categorical(y,num_classes):
	result = []
	for y_ in y:
		tem_y=[0]*num_classes
		tem_y[y_]=1.
		result.append(tem_y)
	return np.array(result)

def encode_y(Y_train_path=Y_train_path):
	"""
	Encodes the labels
	:param Y_train_path: Path to labels in BIES format
	:return: Array of one hot encoded training labels
	"""
	#Training Labels
	with open(Y_train_path, 'r', encoding='utf8') as f:
			label_original_lines = f.readlines()
	Y_tra = [convert_labels_to_integer(label) for label in label_original_lines]

	#One Hot Encoding of Training Labels
	for y in Y_tra:
		Y_train.append(my_to_categorical(y,num_classes=4))
	return Y_train


def pad_sequences(y,max_len):
	lengths = []
	for y_ in y:
		lengths.append(len(y_))
	if max_len == None:
		max_len = np.max(lengths)
	num_samples = len(y)
	sample_shape=tuple()
	for y_ in y:
		if not len(y_):
			continue
		sample_shape = np.asarray(y_).shape[1:]
		break
	result = (np.ones((num_samples,max_len)+sample_shape) * 0).astype('int32')

	for i,y_ in enumerate(y):
		if not len(y_):
			continue
		result_tmp = y_[:max_len]
		result_tmp = np.asarray(result_tmp,dtype='int32')
		result[i,:len(result_tmp)] = result_tmp
	return result

def pad_data(X_train_uni,Y_train):
	"""
	Pad training set sequences
	:param X_train_path: Path to X encoded
	:param Y_train_path: Path to Y encoded
	:return: padded training sets
	"""
	max_len = (sum([len(line) for line in X_train_uni]) / len(X_train_uni))
	MAX_LEN = round(max_len)+1
	train_x_uni_padded = pad_sequences(X_train_uni,MAX_LEN)
	train_y_padded = pad_sequences(Y_train,MAX_LEN)
	return train_x_uni_padded,train_y_padded

def precision(y_true, y_pred):
	"""Precision metric.
	Only computes a batch-wise average of precision.
	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())

	return precision

def bilstm_model(inputs,reuse=False):
	LEN = 2000000
	with tf.variable_scope('build_mode',reuse=reuse):
		initializer = tf.contrib.layers.xavier_initializer()
		embed_matrix = tf.get_variable('embedding',shape=[LEN,64],initializer=initializer)
		embedding_input = tf.nn.embedding_lookup(embed_matrix,inputs)
		def lstm_cell_fw():
			return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(lstm_dim),output_keep_prob=0.8)
		def lstm_cell_bw():
			return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(lstm_dim),output_keep_prob=0.8)
		with tf.variable_scope('rnn_layer',reuse=False):
			stacked_Lstm_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw() for _ in range(2)])
			stacked_Lstm_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw() for _ in range(2)])
			rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(stacked_Lstm_fw,stacked_Lstm_bw,embedding_input,dtype=tf.float32,time_major=False)

		fully_layer_inputs = tf.reshape(tf.concat(rnn_outputs,axis=-1),[-1,2*lstm_dim])

		with tf.variable_scope('fully_layer',reuse=False):
			fully_weights = tf.get_variable('fully_weigthes',shape=[2*lstm_dim,num_class],initializer=initializer)
			logits = tf.matmul(fully_layer_inputs,fully_weights)
	return logits

if __name__ == '__main__':
	input = tf.placeholder(tf.int32,shape=[Batch_size,None])
	label = tf.placeholder(tf.int32,shape=[Batch_size,None,num_class])
	word2index()
	X_train_uni = tokenize_dataset()
	Y_train = encode_y()

	train_x_uni_padded,train_y_padded = pad_data(X_train_uni,Y_train)
	train_x_uni_padded = shuffle(train_x_uni_padded)
	train_y_padded = shuffle(train_y_padded)
	total_batch = int(len(train_x_uni_padded)/Batch_size)
	logits = bilstm_model(input)
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits))
	with tf.variable_scope('opt',reuse=None):
		opt = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
	y_predict = tf.argmax(logits,axis=1)
	correct_prediction = tf.equal(y_predict,tf.argmax(tf.reshape(label,[-1,num_class])))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	saver = tf.train.Saver()
	with tf.Session() as sess:
		#sess.run(tf.initialize_all_variables())
		sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())
		print("Training")
		for epoch in range(20):
			avg_cost = 0.0
			#print('total_batch is',total_batch)
			for i in range(total_batch):
				input_batch = train_x_uni_padded[i*Batch_size:(i+1)*Batch_size]
				real_label = train_y_padded[i*Batch_size:(i+1)*Batch_size]
				feed = {input:input_batch,label:real_label}
				_,c = sess.run([opt,cross_entropy],feed_dict=feed)
				avg_cost += c/total_batch
				print('loss is',c)
			print("avg cost in the training phase epoch {}: {}".format(epoch, avg_cost))
			saver.save(sess,'./model.ckpt')
