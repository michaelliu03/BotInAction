#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
from datetime import datetime
import tensorflow as tf

class TextCNN():
    def __init__(self,seq_length,num_classes,vocab_size):
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size



