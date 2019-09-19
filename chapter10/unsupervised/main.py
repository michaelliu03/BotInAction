#!/usr/bin/env python
#-*-coding:utf-8-*-

import json
from tfidf import *

corpus_filepath = u"../../data/chapter10/AnyQ_dataset_new_2.json"
stop_words_filepath = u"stop_words.utf8"

def bot_process(inputfilepath):
    qaList = []
    with open(inputfilepath, 'r', encoding='utf-8')as fp:
        content = fp.readlines()
        for line in content:
            dict = json.loads(line)
            qaList.append(dict)
    #print(qaList)
    tfmodel = TfidfModel(qaList, stop_words_filepath, None, 0)


if __name__ == '__main__':
   bot_process(corpus_filepath)