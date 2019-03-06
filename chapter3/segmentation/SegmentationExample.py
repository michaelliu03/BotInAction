#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:SegmentationExample.py
# @Author: Michael.liu
# @Date:2019/3/6
# @Desc: NLP Segmentation


import jieba
from pyhanlp import *

import os
import codecs


testCase = [
    "我正在撰写实战对话机器人",
    "王总和小丽结婚了",
    "柯文哲纠正司仪“恭请市长”说法：别用封建语言",
    "签约仪式前，秦光荣、李纪恒、仇和等一同会见了参加签约的企业家。",
    "王国强、高峰先生和汪洋女士、张朝阳光着头、韩寒、小四",
    "张浩和胡健康复员回家了",
    "编剧邵钧林和稽道青说",
    "这里有关天培的有关事迹",
    "龚学平等领导,邓颖超生前",
    "微软的比尔盖茨、Facebook的扎克伯格跟桑德博格",
]

jiebaSegList = []
hanlpSegList =[]
def jiebaSeg():
    for sentence in testCase:
        seg_list = jieba.cut(sentence)
        jiebaSegList.extend(seg_list)
        #print( "/".join(seg_list))
    #return jiebaSegList

def pyHanlpSeg():
    for sentence in testCase:
       seg_list = HanLP.segment("".join(sentence))
       for item in seg_list:
           hanlpSegList.append(item.word)

if __name__ == '__main__':
    print("jieba seg")
    jiebaSeg()
    print("Hanlp Seg")
    pyHanlpSeg()

    sameNum=0
    diffNum=0
    sameList = []
    diffList = []
    # 比较两种切词的异同
    for i in jiebaSegList:
        for j in hanlpSegList:
            if i == j:
               sameNum +=1
               sameList.append(i)

    for b in (jiebaSegList + hanlpSegList):
        if b not in sameList:
            diffList.append(b)


    print(sameList)
    print(diffList)











