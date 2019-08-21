#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:max_probability_seg.py
# @Author: Michael.liu
# @Date:2019/3/19
# @Desc: NLP Segmentation ToolKit - Hanlp Python Version

import re
import csv
import xlwt
import jieba
import jieba.posseg as pseg
from collections import Counter

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
    return stopwords

def remstopwords(sentences):
    stopwords = stopwordslist('../../../data/chapter4/hmm_pos/stopwords.txt')
    outstr = ''
    for word in sentences:
        if word not in stopwords:
            if word !='\t' and '\n':
                outstr +=word
    return outstr

def seg(input_str):
    sentence_seged = jieba.posseg.cut(input_str.strip())
    outstr = ''
    for x in sentence_seged:
        outstr += "{}/{},".format(x.word, x.flag) + " "
        #outstr += remstopwords(x.word) + " "
    return outstr




