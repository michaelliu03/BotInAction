#!/usr/bin/env python
#-*-coding:utf-8-*-
import codecs
import numpy as np

trainfilepath = "../../data/chapter4/example2/train/train.txt"
testfilepath = "../../data/chapter4/example2/test/test.txt"



def read_train(path):
    #train_corpus_data =[]
    #train_corpus_label=[]
    train_data = codecs.open(path, 'r', encoding='utf-8')
    return  train_data


def load_train_data(maxlen,min_count,trainpath):
    train_data = []
    train_label= []
    _data = read_train(trainfilepath)
    for item in _data:
        train_data.append(item.split("\t"))[1]

    for item2 in train_data:
      print(item2)


load_train_data(200,4,trainfilepath)