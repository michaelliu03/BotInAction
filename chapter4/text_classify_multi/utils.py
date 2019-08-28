#!/usr/bin/env python
#-*-coding:utf-8-*-
import codecs
import re
import jieba
import string
import pandas as pd


trainfilepath = "../../data/chapter4/example2/train/train.txt"
testfilepath = "../../data/chapter4/example2/test/test.txt"

# 加载停用词
with open("../stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()


def tokenize_text(text):
    filtered_tokens =[]
    tokens = jieba.cut(text)
    tokens = [replace_num(token.strip())  for token in tokens]
    for token in tokens:
        if token  not in stopword_list and len(token) > 1 :
           filtered_tokens.append(token)
    #filtered_tokens = [token for token in tokens if token not in stopword_list]
    #filtered_text = ''.join(filtered_tokens)
    return filtered_tokens


def remove_special_characters(tokens):
    #tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(tokens):
    #tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text

def read_train(path):
    #train_corpus_data =[]
    #train_corpus_label=[]
    train_data = codecs.open(path, 'r', encoding='utf-8')
    return  train_data

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def replace_num(string):
    if(is_number(string)):
        string = '*'
    return string



def load_train_data(maxlen,min_count,trainpath):
    all_ = []
    train_data = []
    train_label= []
    _data = read_train(trainfilepath)
    for item in _data:
        text = item.split("\t")[1]
        text = tokenize_text(text)

        train_data.append(text)
        train_label.append(item.split("\t")[0])

    abc = pd.Series(train_data).value_counts()
    # abc = abc[abc >= min_count]
    # abc[:] = range(1, len(abc) + 1)
    # vocabulary_size = len(abc)
    # print('vocabulary_size')
    # print(vocabulary_size)
    # abc[''] = 0

        #train_data = train_data(lambda s: replace_num(s))
    #all_[0] = train_data
    for item in train_data:
        print(item)
    #for item2 in train_data:
    #  print(item2)


load_train_data(200,4,trainfilepath)