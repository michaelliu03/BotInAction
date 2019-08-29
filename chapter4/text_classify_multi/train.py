#!/usr/bin/env python
#-*-coding:utf-8-*-
from __future__  import  print_function

import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import  tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from chapter4.text_classify_multi.utils import *
from chapter4.text_classify_multi.attention import attention


trainfilepath = "../../data/chapter4/example2/train/train.xlsx"
testfilepath = "../../data/chapter4/example2/test/test.xlsx"

# Read parameters
training_config = sys.argv[1]
params = json.loads(open(training_config).read())

embed_dim = params['embedding_dim']
hidden_size = params['hidden_size']
keep_prob = params['dropout_keep_prob']
attention_size = params['attention_size']
maxlength = params['maxlength']
l2_reg_lambda = params['l2_reg_lambda']
batch_size = params['batch_size']
num_epochs = params['num_epochs']
min_count = params['min_count']

(X_train,y_train),vocabulary_size,sequence_length =  load_train_data(maxlength,min_count,trainfilepath)

batch_ph = tf.placeholder(tf.int32,[None,sequence_length])
target_ph = tf.placeholder(tf.float32,[None,y_train.shape[1]])
seq_len_ph = tf.placeholder(tf.int32,[None])
keep_prob_ph = tf.placeholder(tf.float32)

#Embedding layer
embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size,embed_dim],-1.0,1.0),trainable=True)
batch_embedded = tf.nn.embedding_lookup(embeddings_var,batch_ph)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,keep_prob)

rnn_outputs, _ = rnn(lstm_cell, inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)

# Attention layer
attention_output, alpha = attention(rnn_outputs, attention_size)