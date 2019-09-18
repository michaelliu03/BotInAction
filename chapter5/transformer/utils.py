#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import numpy as np
import collections

def read_file(file_name):
    """
    读取数据文件
    :param file_name:
    :return:
    """
    contents, labels = [], []
    with open(file_name,'r',encoding='utf-8') as f:
        for line in f:
            try:
                label, content = line.strip().split("\t")
                if content:
                    contents.append(content)
                    labels.append(label)
            except Exception as e:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """
    根据训练数据构建词汇表并存为 txt 文件
    :param train_dir: 训练数据路径
    :param vocab_dir: 词汇表存储路径
    :param vocab_size: 词汇表大小
    :return:
    """
    data_train, _ = read_file(train_dir)

    all_data = []
    # 将字符串转为单个字符的list
    for content in data_train:
        for word in content:
            if word.strip():
                all_data.append(word)

    counter = collections.Counter(all_data)
    counter_pairs = counter.most_common(vocab_size - 2)
    words, _ = list(zip(*counter_pairs))
    words = ['<UNK>'] + list(words)
    words = ['<PAD>'] + list(words)

    with open(vocab_dir, "a",encoding='utf-8') as f:
        f.write('\n'.join(words) + "\n")

    return 0


def word_2_id(vocab_dir):
    """
    :param vocab_dir:
    :return:
    """
    with open(vocab_dir,'r',encoding='utf-8') as f:
        words = [_.strip() for _ in f.readlines()]

    word_dict = {}
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def cat_2_id():
    """
    :return:
    """
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    id_to_cat = dict((v, k) for k, v in cat_to_id.items())

    return cat_to_id, id_to_cat


def process_file(data_dir, word_to_id, cat_to_id, seq_length=512):
    """
    :param data_dir:
    :param word_to_id:
    :param cat_to_id:
    :param seq_length:
    :return:
    """
    contents, labels = read_file(data_dir)

    data_id, label_id = [], []
    for i in range(len(contents)):
        sent_ids = [word_to_id.get(w) if w in word_to_id else word_to_id.get("<UNK>") for w in contents[i]]
        # pad to the required length
        if len(sent_ids) > seq_length:
            sent_ids = sent_ids[:seq_length]
        else:
            padding = [0] * (seq_length - len(sent_ids))
            sent_ids += padding
        data_id.append(sent_ids)
        y_pad = [0] * len(cat_to_id)
        y_pad[cat_to_id[labels[i]]] = 1
        label_id.append(y_pad)

    return np.array(data_id), np.array(label_id)


def batch_iter(x, y, batch_size=32, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[shuffle_indices]
        y_shuffle = y[shuffle_indices]
    else:
        x_shuffle = x
        y_shuffle = y
    for i in range(num_batch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)
        yield (x_shuffle[start_index:end_index], y_shuffle[start_index:end_index])


@staticmethod
def trans_to_index(inputs, word_to_index):
    """
    将输入转化为索引表示
    :param inputs: 输入
    :param word_to_index: 词汇-索引映射表
    :return:
    """
    inputs_idx = [[word_to_index.get(word, word_to_index["<UNK>"])for word in sentence] for sentence in inputs]

    return inputs_idx

@staticmethod
def trans_label_to_index(labels, label_to_index):
    """
    将标签也转换成数字表示
    :param labels: 标签
    :param label_to_index: 标签-索引映射表
    :return:
    """
    labels_idx = [label_to_index[label] for label in labels]
    return labels_idx

def padding(self, inputs, sequence_length):
    """
    对序列进行截断和补全
    :param inputs: 输入
    :param sequence_length: 预定义的序列长度
    :return:
    """
    new_inputs = [sentence[:sequence_length]
                  if len(sentence) > sequence_length
                  else sentence + [0] * (sequence_length - len(sentence))
                  for sentence in inputs]

    return new_inputs

def data_help(ham_data_path,spam_data_path,save_path):
    inputs = []
    with open(ham_data_path, "r", encoding="utf8") as fr:
        for line in fr.readlines():
            print(line)

            # try:
            #     text, label = line.strip().split("<SEP>")
            #     inputs.append(text.strip().split(" "))
            #     labels.append(label)
            # except:
            #     continue

if __name__ == '__main__':
    print("this is begin")