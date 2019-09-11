#!/usr/bin/env python
#-*-coding:utf-8-*-
import tensorflow as tf

from chapter5.FastText.utils import  *
def test_model(sess, graph, x_, y_):
    """
    :param sess:
    :param graph:
    :param x_:
    :param y_:
    :return:
    """
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 64)
    total_loss = 0.0
    total_acc = 0.0

    input_x = graph.get_operation_by_name('input_x').outputs[0]
    input_y = graph.get_operation_by_name('input_y').outputs[0]
    loss = graph.get_operation_by_name('loss/loss').outputs[0]
    acc = graph.get_operation_by_name('accuracy/acc').outputs[0]

    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {input_x: x_batch, input_y: y_batch}
        test_loss, test_acc = sess.run([loss, acc], feed_dict=feed_dict)
        total_loss += test_loss * batch_len
        total_acc += test_acc * batch_len

    return total_loss / data_len, total_acc / data_len

def test(vocab_dir,test_dir,max_length):
    word_to_id, id_to_word = word_2_id(vocab_dir)
    cat_to_id, id_to_cat = cat_2_id()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, max_length)
    graph_path = "./model/chapter5/fast/model-40000.meta"
    model_path = "./model/chapter5/fast"
    graph = tf.Graph()
    saver = tf.train.import_meta_graph(graph_path, graph=graph)
    sess = tf.Session(graph=graph)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    test_loss, test_acc = test_model(sess, graph, x_test, y_test)
    print("Test loss: %f, Test acc: %f" %(test_loss, test_acc))