#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Segment.py
# @Author: Michael.liu
# @Date:2019/2/12

import numpy as np
import os ,time,sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf  import viterbi_decode
#from data import pad_sequences,batch_yield



class BiLSTM_CRF(object):
    def __init__(self,args,embedding,tag2label,vocab,paths,config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch