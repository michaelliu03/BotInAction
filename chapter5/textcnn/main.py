#!/usr/bin/env python
#-*-coding:utf-8-*-
import os

from utils import  *
from train import *

if __name__ == '__main__':
    print("begin train....")
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, vocab_size)

    main()
