#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Segment.py
# @Author: Michael.liu
# @Date:2019/2/12

import numpy as np
from scipy  import  stats
import matplotlib.pyplot as plt
import math


# 二项分布
def binomial(_n,_p,_k):
    n = _n
    p = _p
    k = np.arange(0, _k)
    print(k)

    print("*" * 20)

    binomial = stats.binom.pmf(k, n, p)
    print(binomial)


    plt.plot(k, binomial, 'o-')
    plt.title('binomial:n=%i,p=%.2f' % (n, p), fontsize=15)
    plt.xlabel('number of success')
    plt.ylabel('probalility of success', fontsize=15)
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
   binomial(20, 0.5, 41)
