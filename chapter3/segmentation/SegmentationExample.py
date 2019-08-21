#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:SegmentationExample.py
# @Author: Michael.liu
# @Date:2019/3/6
# @Desc: NLP Segmentation


import jieba
from jpype import *

import jpype
from  jpype import *
import os


a=u'D:\\DevProgram\\Java\\jdk1.8.0_211\\jre\\bin\\server\\jvm.dll'    #jvm.dll启动成功
jarpath = os.path.abspath("D:\\liepin_project\\seg_lib\\ins-segmentation-0.0.1-jar-with-dependencies.jar")
#jarpath = os.path.join(os.path.abspath('D:\\liepin_project\\seg_lib\\ins-segmentation-0.0.1-jar-with-dependencies.jar'), a)#第二个参数是jar包的路径
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" %(jarpath))#启动jvm

#jpype.startJVM(a, "-Djava.class.path=D:\\liepin_project\\seg_lib\\ins-segmentation-0.0.1-jar-with-dependencies.jar")
HanLP = JClass('com.hankcs.hanlp.HanLP')

testCases = [
    "商品和服务",
    "结婚的和尚未结婚的确实在干扰分词啊",
    "买水果然后来世博园最后去世博会",
    "中国的首都是北京",
    "欢迎新老师生前来就餐",
    "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
    "随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。"]



jiebaSegList = []
hanlpSegList =[]
def jiebaSeg():
    for sentence in testCases:
        seg_list = jieba.cut(sentence)
        jiebaSegList.extend(seg_list)
        #print( "/".join(seg_list))
    #return jiebaSegList

def pyHanlpSeg():
    for sentence in testCases:
       seg_list = HanLP.newSegment().seg("".join(sentence))
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







jpype.shutdownJVM()



