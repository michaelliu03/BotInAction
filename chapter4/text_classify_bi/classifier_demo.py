#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Segment.py
# @Author: Michael.liu
# @Date:2019/2/12
# @Desc: NLP Segmentation ToolKit - Hanlp Python Version


from chapter4.text_classify_bi.feature_extraction import bow_extractor, tfidf_extractor, word2vector_extractor
from chapter4.text_classify_bi.util import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from chapter4.text_classify_bi.normalization import normalize_corpus

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import warnings

warnings.filterwarnings("ignore")

def get_data():
    hamdata, hamlabel = get_ham_data(PosPath)
    spamdata,spamlabel = get_spam_data(NegPath)
    # 构成相关的corpus
    corpus_data = hamdata + spamdata   # 语料相加
    corpus_label = hamlabel + spamlabel # 标签相加
    return corpus_data, corpus_label

def prepare_datasets(corpus, labels, test_data_proportion =0.3):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels,
                                                        test_size=test_data_proportion, random_state=42)
    return train_X, test_X, train_Y, test_Y

def get_metrics(true_labels, predicted_labels):
    print('准确率:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        2))
    print('精度:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        2))
    print('召回率:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        2))
    print('F1得分:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        2))

def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions

def classifer_type(personal):
    if personal == 1:
       mnb = MultinomialNB()
       classifertype  = mnb
    # if personal == 2:
    #    svm = SGDClassifier(loss='hinge', n_iter=50)
    #    classifertype = svm
    if personal == 3:
       lr = LogisticRegression()
       classifertype = lr
    return classifertype


# 通过 bag of words 进行
def bow_extractor_model(norm_train_corpus,norm_test_corpus,train_labels,test_labels,type):
    classifertype = classifer_type(type)
    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)
    bow_predictions = train_predict_evaluate_model(classifier=classifertype,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)


# 通过 tf-idf 进行
def tf_idf_model(norm_train_corpus,norm_test_corpus,train_labels,test_labels,type):
    classifertype = classifer_type(type)
    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)
    tfidf_predictions = train_predict_evaluate_model(classifier=classifertype,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)

# 通过 word2vector 进行
def word2vector_model(norm_train_corpus,norm_test_corpus,train_labels,test_labels,type):
    #print("to be continue!!!")
    word2vector_extractor(norm_train_corpus)

def process():
    corpus, labels = get_data()  # 获取数据集
    print("总的数据量:", len(labels))
    corpus, labels = remove_empty_docs(corpus, labels)
    label_name_map = ["垃圾邮件", "正常邮件"]

    # 对数据进行划分
    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,
                                                                            labels,
                                                                          test_data_proportion=0.3)

    # 进行归一化
    norm_train_corpus = normalize_corpus(train_corpus)
    norm_test_corpus = normalize_corpus(test_corpus)
    print("基于词袋模型分类器")
    bow_extractor_model(norm_train_corpus,norm_test_corpus,train_labels,test_labels, 1)
    #bow_extractor_model(norm_train_corpus, norm_test_corpus, train_labels, test_labels, 2)
    bow_extractor_model(norm_train_corpus, norm_test_corpus, train_labels, test_labels, 3)
    #word2vector_model(norm_train_corpus,norm_test_corpus,train_labels,test_labels, 1)
    print("基于tf-idf分类器")
    tf_idf_model(norm_train_corpus, norm_test_corpus, train_labels, test_labels, 1)
    #tf_idf_model(norm_train_corpus, norm_test_corpus, train_labels, test_labels, 2)
    tf_idf_model(norm_train_corpus, norm_test_corpus, train_labels, test_labels, 3)

if  __name__ == "__main__":
    process()