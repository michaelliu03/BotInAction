#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 52nlpcn@gmail.com
# Copyright 2014 @ YuZhen Technology
#
# 4 tags for character tagging: B(Begin), E(End), M(Middle), S(Single)

import codecs
import sys

def character_split(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        for word in line.strip():
            word = word.strip()
            if word:
                output_data.write(word + "\tB\n")
        output_data.write("\n")
    input_data.close()
    output_data.close()

input_file=r'E:\self-project\BotInAction\data\chapter3\msr_test.utf8'
output_file=r'E:\self-project\BotInAction\data\chapter3\msr_test_res.utf8'
if __name__ == '__main__':
    character_split(input_file, output_file)