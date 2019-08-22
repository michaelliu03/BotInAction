#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Segment.py
# @Author: Michael.liu
# @Date:2019/2/12
import  jieba
import re
import pandas as pd
import codecs
import string

corpus_path = u"../../data/chapter4/example3/lda_corpus.txt"
corpus_save_path = u"../../data/chapter4/example3/lda_corpus_seg.csv"

with open("../stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()

def tokenize_text(text):
    tokens = jieba.lcut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:
        text =" ".join(jieba.lcut(text))
        normalized_corpus.append(text)

    return normalized_corpus

def load_corpus_data():
    corpus_data = codecs.open(corpus_path,'r',encoding='utf-8')  # 读取文件
    # book_titles = book_data['title'].tolist()
    # book_content = book_data['content'].tolist()
    return corpus_data

def process():
    book_data = load_corpus_data()
    norm_book_content = normalize_corpus(book_data)
    with open(corpus_save_path,'w',encoding='utf-8') as fw:
        for item in norm_book_content:
           fw.write(item)
    fw.close()
        #print(norm_book_content)


if __name__ =="__main__":
    process()