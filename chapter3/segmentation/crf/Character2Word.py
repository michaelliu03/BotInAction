#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Character2Word.py
# @Author: Michael.liu
# @Date:2019/3/12
# @Desc: NLP Segmentation ToolKit - Hanlp Python Version

import codecs


def Character2Word(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    i=0
    for line in input_data.readlines():
        if line == "\n" or line.strip()=='':
            output_data.write("\n")
        else:
            char_tag_pair = line.strip().split('\t')
            char = char_tag_pair[0]
            tag = char_tag_pair[2]
            if tag == 'B':
                output_data.write(' ' + char)
            elif tag == 'M':
                output_data.write(char)
            elif tag == 'E':
                output_data.write(char + ' ')
            else: # tag == 'S'
                output_data.write(' ' + char + ' ')
    input_data.close()
    output_data.close()

input_file=r'E:\self-project\BotInAction\data\chapter3\result.txt'
output_file=r'E:\self-project\BotInAction\data\chapter3\msr_res_res.utf8'
if __name__ == '__main__':
    Character2Word(input_file, output_file)