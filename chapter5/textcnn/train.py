#!/usr/bin/env python
#-*-coding:utf-8-*-
import os,sys
from datetime import datetime
import tensorflow as tf
import json

from utils import  *
from test import *

# Read parameters
training_config = u"D:\\liuyu\\桌面\\git\\BotInAction\\chapter5\\textcnn\\training_config.json"#sys.argv[1]
params = json.loads(open(training_config).read())

embed_dim = params['embedding_dim']
max_length=params['max_length']
num_epochs = params['num_epochs']
vocab_size = params['vocab_size']
num_classes=params['num_classes']
num_filters= params['num_filters']
hidden_dim = params['hidden_dim']
seq_length=params['seq_length']
learning_rate=params['learning_rate']
test_train_ratio = params['test_train_ratio']

base_dir="../../data/chapter5/fasttext/"
textcnn_dir= "../../data/chapter5/textcnn/"
train_dir=os.path.join(base_dir, 'cnews.train.txt')
test_dir=os.path.join(base_dir,'cnews.test.txt')
val_dir=os.path.join(base_dir, 'cnews.val.txt')
vocab_dir=os.path.join(textcnn_dir, 'cnews.vocab.txt')

model_save = "../../model/chapter5/textcnn/model"
grap_path ="../../model/chapter5/textcnn/model-7000.meta"
model_path="../../model/chapter5/textcnn"


class TextCNN():
    def __init__(self):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.filter_sizes = [3,4,5]
        self.embedding_dim = embed_dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.input_x = tf.placeholder(tf.int32,[None,self.seq_length],name='input_x')
        self.input_y = tf.placeholder(tf.float32,[None,self.num_classes],name='input_y')
        self.drop_prob =tf.placeholder(tf.float32,name='drop_prob')
        self.learning_rate=tf.placeholder(tf.float32,name='learn_rate')
        self.l2_loss = tf.constant(0.0)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale =0.01)
        self.inference()

    def inference(self):
        with tf.name_scope("embedding"):
            embedding = tf.get_variable("embedding",[self.vocab_size,self.embedding_dim])
            embedding_inputs= tf.nn.embedding_lookup(embedding,self.input_x)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-%s" % i):
                # conv layer
                conv = tf.layers.conv1d(embedding_inputs, self.num_filters, filter_size,
                                        padding='valid', activation=tf.nn.relu,
                                        kernel_regularizer=self.regularizer)
                # global max pooling
                pooled = tf.layers.max_pooling1d(conv, self.seq_length - filter_size + 1, 1)
                pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 2)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            h_drop = tf.layers.dropout(h_pool_flat, self.drop_prob)

        with tf.name_scope("score"):
            fc = tf.layers.dense(h_pool_flat, self.hidden_dim, activation=tf.nn.relu, name='fc1')
            fc = tf.layers.dropout(fc, self.drop_prob)
            # classify
            self.logits = tf.layers.dense(fc, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)

            l2_loss = tf.losses.get_regularization_loss()
            self.loss = tf.reduce_mean(cross_entropy, name="loss")
            self.loss += l2_loss

            # optim
            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="acc")


def evaluate(sess, model, x_, y_):
    """
    评估 val data 的准确率和损失
    """
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 64)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {model.input_x: x_batch, model.input_y: y_batch,
                     model.drop_prob: 0}
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def main():
    word_to_id, id_to_word = word_2_id(vocab_dir)
    cat_to_id, id_to_cat = cat_2_id()

    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, max_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, max_length)

    epochs = 5
    best_acc_val = 0.0  # 最佳验证集准确率
    train_steps = 0
    val_loss = 0.0
    val_acc = 0.0
    with tf.Graph().as_default():
        # seq_length = 512
        # num_classes = 10
        # vocab_size = 5000
        cnn_model = TextCNN()
        saver = tf.train.Saver()
        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                print('Epoch:', epoch + 1)
                batch_train = batch_iter(x_train, y_train, 32)
                for x_batch, y_batch in batch_train:
                    train_steps += 1
                    learn_rate = 0.001
                    # learning rate vary
                    feed_dict = {cnn_model.input_x: x_batch, cnn_model.input_y: y_batch,
                                 cnn_model.drop_prob: 0.5, cnn_model.learning_rate: learn_rate}

                    _, train_loss, train_acc = sess.run([cnn_model.optim, cnn_model.loss,
                                                         cnn_model.acc], feed_dict=feed_dict)

                    if train_steps % 1000 == 0:
                        val_loss, val_acc = evaluate(sess, cnn_model, x_val, y_val)

                    if val_acc > best_acc_val:
                        # 保存最好结果
                        best_acc_val = val_acc
                        last_improved = train_steps
                        saver.save(sess, model_save, global_step=train_steps)
                        # saver.save(sess=session, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    now_time = datetime.now()
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(train_steps, train_loss, train_acc, val_loss, val_acc, now_time, improved_str))

if __name__ =='__main__':
    print("begin train...")
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir,vocab_dir,vocab_size)
    main()
    #test(vocab_dir,test_dir,max_length,grap_path,model_path)
