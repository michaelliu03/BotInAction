#!/usr/bin/env python
#-*-coding:utf-8-*-

import json
from tfidf import *

corpus_filepath = u"../../data/chapter10/AnyQ_dataset_new_2.json"


def bot_process(inputfilepath):
    with open(inputfilepath, 'r', encoding='utf-8')as fp:
        content = fp.readlines()
        print(type(content))
        #s = json.dumps(content, ensure_ascii=False)
        #dict = json.loads(s)
        #print(content)
        #print(s)
        # content = json.load(fp)
        #content = fp.readlines()
        #dict = json.dumps(content)
        #dict = json.loads(content)
        #print(content)
    #tfmodel = TfidfModel(dict,None,None,0)
    #tfmodel.get_top_n_answer("上飞机可以带充电宝吗", n=15, interval=0.2, answer_num=5)



if __name__ == '__main__':
   #load_data(corpus_filepath)
   bot_process(corpus_filepath)