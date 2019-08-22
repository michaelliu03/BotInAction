#!/usr/bin/env python
#-*-coding:utf-8-*-

corpus_path = u"../../data/chapter4/example4/jobtitle_title_JD_seg.txt"
def read_corpus(path):
    with open(path, encoding="utf8") as f:
        sent = f.readlines()
        for i in sent:
            print(i)
    return sent


#read_corpus(corpus_path)