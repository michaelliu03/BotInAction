#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Segment.py
# @Author: Michael.liu
# @Date:2019/2/12


import tensorflow as tf
import numpy as np
import  os , argparse,time,random
from model import BiLSTM_CRF