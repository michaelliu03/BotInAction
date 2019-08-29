#!/usr/bin/env python
#-*-coding:utf-8-*-
import codecs
import re
import jieba
import string
import pandas as pd
import json
import numpy as np


trainfilepath = "../../data/chapter4/example2/train/train.xlsx"
testfilepath = "../../data/chapter4/example2/test/test.xlsx"

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

#输出词表
def output_vocabulary(abc):
    values = abc[:]
    keys = abc.index
    vocabulary = dict(zip(keys,values))
    with open('../../model/chapter4/example2/vocabulary.json','w') as outfile:
        json.dump(vocabulary,outfile,indent=4)

def output_labels(labeels):
    with open('../../model/chapter4/example2/label.json','w') as outfile:
        json.dump(labeels,outfile,indent=4)



def load_train_data(maxlen,min_count,trainpath):
    all_ = pd.read_excel(trainpath, header=None)
    all_['words'] = all_[1].apply(lambda s: list(tokenize_text(s)))
    #print(all_.head(5))
    print(all_['words'].head(5))

    content = []
    for i in all_['words']:
        content.extend(i)

    #词表
    abc = pd.Series(content).value_counts()
    #print(abc)
    abc = abc[abc >= min_count]
    abc[:]  = range(1,len(abc)+1)
    vocabulary_size = len(abc)
    print('vocabulary_size')
    print(vocabulary_size)
    abc[''] =0

    output_vocabulary(abc)

    def doc2num(s,maxlen):
        s = [i for i in s if i in abc.index]
        s = s[:maxlen-1] + ['']* max(1,maxlen-len(s))
        return list(abc[s])

    print('one hot conversion')
    all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s,maxlen))
    x_train = np.array(list(all_['doc2num']))
    #print(x_train)
    labels = sorted(list(set(all_[0])))
    output_labels(labels)
    num_labels = len(labels)
    one_hot = np.zeros((num_labels,num_labels),int)
    np.fill_diagonal(one_hot,1)
    label_dict = dict(zip(labels,one_hot))
    #print(label_dict)
    y_train = np.array(all_[0].apply(lambda y:label_dict[y]).tolist())
    print(y_train)

    return (x_train,y_train),vocabulary_size,maxlen



    #idx = range(len(all_))




    # all_ = []
    # train_data = []
    # train_label= []
    # _data = read_train(trainfilepath)
    # for item in _data:
    #     text = item.split("\t")[1]
    #     text = tokenize_text(text)
    #
    #     train_data.append(list(text))
    #     train_label.append(item.split("\t")[0])

    #abc = pd.Series(train_data).value_counts()
    # abc = abc[abc >= min_count]
    # abc[:] = range(1, len(abc) + 1)
    # vocabulary_size = len(abc)
    # print('vocabulary_size')
    # print(vocabulary_size)
    # abc[''] = 0

        #train_data = train_data(lambda s: replace_num(s))
    #all_[0] = train_data
    #for item in train_data:
    #    print(item)
    #for item2 in train_data:
    #  print(item2)

if __name__ == "__main__":
    load_train_data(200,4,trainfilepath)