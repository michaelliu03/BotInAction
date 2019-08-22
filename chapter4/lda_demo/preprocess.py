#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Segment.py
# @Author: Michael.liu
# @Date:2019/2/12
import re
import os

inputfilepath = ""
out_file_path = ""

def get_corpus_data(dirpath):
    pattern1 = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    for (dirname, dirs, files) in os.walk(dirpath):

        for file in files:
            if file.endswith('.txt'):
                filename = os.path.join(dirname, file)
                print(dirpath + filename)
                with open(filename,'r',encoding='utf-8') as f:
                    for line in f.readlines():
                        if pattern1.findall(line):
                            continue
                        with open(out_file_path, "a",encoding='utf-8') as mom:
                            mom.write(line)
                f.close()
                mom.close()

get_corpus_data(inputfilepath)
