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
def binomial_distribution(_n,_p,_k):
    n = _n
    p = _p
    k = np.arange(0, _k)
    binomial = stats.binom.pmf(k, n, p)
    #print(binomial)

    plt.plot(k, binomial, 'o-')
    plt.title('binomial:n=%i,p=%.2f' % (n, p), fontsize=15)
    plt.xlabel('number of success')
    plt.ylabel('probalility of success', fontsize=15)
    plt.grid(True)

    plt.show()

#泊松分布
def poisson_distribution(n,k1):
    a = np.random.poisson(n,k1)
    plt.hist(a,bins=8,color='g',alpha=0.4,edgecolor='b')
    plt.show()

# 正态分布
def normal_distribution(n,p,k):
    list_d = np.random.normal(n, p, k)
    plt.hist(list_d, bins=8, color='g', alpha=0.4, edgecolor='b')
    plt.show()



if __name__ == "__main__":
    binomial_distribution(20, 0.5, 40)
    poisson_distribution(9,1000)
    normal_distribution(0,1,5000)