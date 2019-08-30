#!/usr/bin/env python
#-*-coding:utf-8-*-
from __future__  import  print_function

import sys,os




import json
import  tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from chapter4.text_classify_multi.utils import *
from chapter4.text_classify_multi.attention import attention


trainfilepath = "../../data/chapter4/example2/train/train.xlsx"
testfilepath = "../../data/chapter4/example2/test/test.xlsx"
training_config_path = u"D:\\liuyu\\桌面\\git\\BotInAction\\chapter4\\text_classify_multi\\training_config.json"

# Read parameters
training_config = training_config_path #sys.argv[1]
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
test_train_ratio = params['test_train_ratio']

(X_train,y_train),(X_test, y_test),vocabulary_size,sequence_length =  load_train_data(maxlength,min_count,trainfilepath,test_train_ratio)

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

# L2 rule
l2_loss = tf.constant(0.0)

drop = tf.nn.dropout(attention_output,keep_prob_ph)
#Fully connected layer
W = tf.Variable(tf.truncated_normal([drop.get_shape()[1].value,y_train.shape[1]],stddev=0.1))

b = tf.Variable(tf.constant(0.1,shape=[y_train.shape[1]]))
y_hat = tf.nn.xw_plus_b(drop,W,b)
l2_loss +=tf.nn.l2_loss(W)
l2_loss +=tf.nn.l2_loss(b)

# define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat,labels = target_ph)) + l2_reg_lambda * l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# Accuracy metric
correct_predictions = tf.equal(tf.argmax(y_hat, 1), tf.argmax(target_ph, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', loss)
# Batch generators
train_batch_generator = batch_generator(X_train, y_train, batch_size)
test_batch_generator = batch_generator(X_test, y_test, batch_size)

delta = 0.5
saver = tf.train.Saver()


print ("x_train dimentions are")
print (X_train.shape[0])
print (X_train.shape[1])

print ("x_test dimentions are")
print (X_test.shape[0])
print (X_test.shape[1])

print ("y_hat are")
print (y_hat.shape[0])
print (y_hat.shape[1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter('./train_track/',sess.graph)
    print("Start learning...")
    for epoch in range(num_epochs):
        loss_train = 0
        loss_test = 0
        accuracy_train = 0
        accuracy_test = 0

        print("epoch: {}\t".format(epoch), end="")
        # Training
        num_batches = X_train.shape[0] // batch_size
        for b in range(num_batches):
            x_batch, y_batch = train_batch_generator.__next__()
            seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
            #   print(seq_len.shape)
            summary_tr, alpha_values, loss_tr, acc, _ = sess.run([merged, alpha, loss, accuracy, optimizer],
                                                                 feed_dict={batch_ph: x_batch, target_ph: y_batch,
                                                                            seq_len_ph: seq_len,
                                                                            keep_prob_ph: keep_prob})
            accuracy_train += acc
            #summary_train += summary_tr
            loss_train = loss_tr * delta + loss_train * (1 - delta)
            #writer.add_summary(summary_tr,epoch*num_batches + b)
        #    print(alpha_values)
        accuracy_train /= num_batches
        #summary_train /= num_batches

        # Validating
        num_batches = X_test.shape[0] // batch_size
        for b in range(num_batches):
            x_batch, y_batch = test_batch_generator.__next__()
            seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
            loss_test_batch, acc = sess.run([loss, accuracy, ], feed_dict={batch_ph: x_batch, target_ph: y_batch,
                                                                           seq_len_ph: seq_len, keep_prob_ph: 1.0})
            accuracy_test += acc
            loss_test += loss_test_batch
        accuracy_test /= num_batches
        loss_test /= num_batches

        #writer.add_summary(summary_tr,epoch)
        #writer.add_summary(accuracy_test,epoch)

        print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
            loss_train, loss_test, accuracy_train, accuracy_test
        ))

        save_path = saver.save(sess, "./model/product_1.ckpt")