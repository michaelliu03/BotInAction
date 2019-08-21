#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Segment.py
# @Author: Michael.liu
# @Date:2019/2/12

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import gensim

def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def word2vector_extractor(corpus):
    tokenized_train = [jieba.lcut(text) for text in corpus]
    #print(tokenized_train[2:10])
    model = gensim.models.Word2Vec(tokenized_train,
                                   size=500,
                                   window=100,
                                   min_count=30,
                                   sample=1e-3)