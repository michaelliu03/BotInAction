#!/usr/bin/env python
#-*-coding:utf-8-*-
from __future__ import print_function
import  sys
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.python.ops import array_ops
import xlwt

from chapter4.text_classify_multi.attention import attention
from chapter4.text_classify_multi.utils import *

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

X_test, y_test, vocabulary_size, sequence_length, x_list, num_labels = load_data_prediction(maxlength, min_count)

model_checkpoint_path = './model/comments_sentiment_model.ckpt'

# Different placeholders
batch_ph = tf.placeholder(tf.int32, [None, sequence_length])
target_ph = tf.placeholder(tf.float32, [None, y_test.shape[1]])
seq_len_ph = tf.placeholder(tf.int32, [None])
keep_prob_ph = tf.placeholder(tf.float32)

# Embedding layer
embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, embed_dim], -1.0, 1.0), trainable=True)
batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

# rnn_outputs, _ = bi_rnn(GRUCell(hidden_size), GRUCell(hidden_size),
#                           inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
rnn_outputs, _ = rnn(GRUCell(hidden_size), inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
# lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
# lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, keep_prob)

# rnn_outputs, _ = bi_rnn(lstm_cell, lstm_cell,
#                           inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
# rnn_outputs, _ = rnn(lstm_cell, inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)

# Attention layer
attention_output, alpha = attention(rnn_outputs, attention_size)

# L2 rule
l2_loss = tf.constant(0.0)
drop = tf.nn.dropout(attention_output, keep_prob_ph)

# output = rnn_outputs[0]
# Fully connected layer
W = tf.Variable(tf.truncated_normal([drop.get_shape()[1].value, y_test.shape[1]], stddev=0.1))

# W = tf.Variable(tf.truncated_normal([hidden_size,y_train.shape[1]], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[y_test.shape[1]]))
y_hat = tf.nn.xw_plus_b(drop, W, b)

y_hat = tf.nn.softmax(y_hat)
predictions = tf.argmax(y_hat, 1)
# Accuracy metric
# accuracy = 1. - tf.reduce_mean(tf.cast(tf.equal(tf.round(y_hat), target_ph), tf.float32))
y_ori = tf.argmax(target_ph, 1)
correct_predictions = tf.equal(predictions, tf.argmax(target_ph, 1))

accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Batch generators
batch_size = 256
train_batch_generator = prediction_batch_generator(X_test, y_test, batch_size)

saver = tf.train.Saver()

print("x_test dimentions are")
print(X_test.shape[0])
print(X_test.shape[1])

print("y_test dimentions are")
print(y_test.shape[0])
print(y_test.shape[1])

# *************************************************************************************************************
# embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, embed_dim], -1.0, 1.0), trainable=True)
# batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)
# batch_size = 256

# W = tf.get_variable('W', [vocabulary_size, embed_dim])
# embedded = tf.nn.embedding_lookup(W, batch_ph)

# inputs = tf.split(embedded, sequence_length, 1)
# **********************************************************************************************************

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.global_variables_initializer())
    accuracy_test = 0
    loss_test = 0
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, model_checkpoint_path)
    labels = json.loads(open('./model/labels_index.json').read())
    wb = xlwt.Workbook()
    ws = wb.add_sheet('question')
    wq = wb.add_sheet('AccRecall')
    predict_labels, pres = [], []
    pres_all, y_all, correct_pres_all = [], [], []
    j = 0
    num_batches = X_test.shape[0] / batch_size
    #    seq_len = np.array([list(x).index(0) + 1 for x in X_test])  # actual lengths of sequences
    for b in range(num_batches):
        x_batch, y_batch = train_batch_generator.next()
        seq_len = np.array([list(x).index(0) + 1 for x in x_batch])
        acc, alpha_all, pres, correct_pres, y = sess.run([accuracy, alpha, predictions, correct_predictions, y_ori],
                                                         feed_dict={batch_ph: x_batch, target_ph: y_batch,
                                                                    seq_len_ph: seq_len, keep_prob_ph: 1.0})
        pres_all.extend(pres)
        y_all.extend(y)
        correct_pres_all.extend(correct_pres)
        accuracy_test += acc
    accuracy_test = accuracy_test / num_batches
    confusion_mat = np.zeros((num_labels, num_labels), float)

    for i in range(0, len(pres_all)):
        confusion_mat[pres_all[i]][y_all[i]] += 1

    for i in range(0, num_labels):
        recall = 0.0
        acc = 0.0
        f1 = 0.0
        if (i == 0):
            ws.write(i, 0, "Category")
            ws.write(i, 1, "Recall")
            ws.write(i, 2, "Accuracy")
            ws.write(i, 3, "F1")
            ws.write(i, 4, "Amount")
        else:
            k = i - 1
            wq.write(i, 0, labels[k])
            if (sum(confusion_mat[k][:]) != 0):
                recall = confusion_mat[k, k] / sum(confusion_mat[k][:])
                acc = confusion_mat[k, k] / axis_sum(confusion_mat, k, num_labels)
                f1 = (2 * recall * acc) / (recall + acc)
            wq.write(i, 1, recall)
            wq.write(i, 2, acc)
            wq.write(i, 3, f1)
            wq.write(i, 4, axis_sum(confusion_mat, k, num_labels))

    print(confusion_mat)

    for prediction in pres_all:
        #        print(prediction)
        predict_labels.append(labels[prediction])
    #    output_attention_weights(alpha)

    for i in range(0, len(pres_all)):
        #    print(j)
        if (j == 0):
            ws.write(j, 0, "Comments")
            ws.write(j, 1, "Original sentiment")
            ws.write(j, 2, "Predicted sentiment")
        else:
            ws.write(j, 0, x_list[i])
            ws.write(j, 1, labels[y_all[i]])
            ws.write(j, 2, labels[pres_all[i]])
        j += 1
    print("acc: {:.3f}".format(accuracy_test))

    #   output_prediction_labels(predict_labels)
    wb.save('./model/gru_comment_sentiment_prediction.xls')
