#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:hmm.py
# @Author: Michael.liu
# @Date:2019/3/19
# @Desc: NLP Segmentation ToolKit - Bot In Action Python Version


import numpy as np

from chapter3.postag.hmm.max_probability_seg import seg


def viterbi(obs_len, states_len, init_p, trans_p, emit_p):
    """
    :param obs_len: 观测序列长度 int
    :param states_len: 隐含序列长度 int
    :param init_p:初始概率 list
    :param trans_p:转移概率矩阵 np.ndarray
    :param emit_p:发射概率矩阵 np.ndarray
    :return:最佳路径 np.ndarray
    """
    max_p = np.zeros((states_len, obs_len))  # max_p每一列为当前观测序列不同隐状态的最大概率
    path = np.zeros((states_len, obs_len))  # path每一行存储上max_p对应列的路径

    # 初始化max_p第1个观测节点不同隐状态的最大概率并初始化path从各个隐状态出发
    for i in range(states_len):
        max_p[i][0] = init_p[i] * emit_p[i][0]
        path[i][0] = i

    # 遍历第1项后的每一个观测序列，计算其不同隐状态的最大概率
    for obs_index in range(1, obs_len):
        new_path = np.zeros((states_len, obs_len))
        # 遍历其每一个隐状态
        for hid_index in range(states_len):
            # 根据公式计算累计概率，得到该隐状态的最大概率
            max_prob = -1
            pre_state_index = 0
            for i in range(states_len):
                each_prob = max_p[i][obs_index - 1] * trans_p[i][hid_index] * emit_p[hid_index][obs_index]
                if each_prob > max_prob:
                    max_prob = each_prob
                    pre_state_index = i

            # 记录最大概率及路径
            max_p[hid_index][obs_index] = max_prob
            for m in range(obs_index):
                # "继承"取到最大概率的隐状态之前的路径（从之前的path中取出某条路径）
                new_path[hid_index][m] = path[pre_state_index][m]
            new_path[hid_index][obs_index] = hid_index
        # 更新路径
        path = new_path

    # 返回最大概率的路径
    max_prob = -1
    last_state_index = 0
    for hid_index in range(states_len):
        if max_p[hid_index][obs_len - 1] > max_prob:
            max_prob = max_p[hid_index][obs_len - 1]
            last_state_index = hid_index
    return path[last_state_index]



def cal_hmm_matrix(observation):
    # 得到所有标签
    word_pos_file = open('../../../data/chapter3/hmm_pos/ChineseDic.txt','r',encoding='utf-8').readlines()
    tags_num = {}
    for line in word_pos_file:
        word_tags = line.strip().split(',')[1:]
        for tag in word_tags:
            if tag not in tags_num.keys():
                tags_num[tag] = 0
    tags_list = list(tags_num.keys())

    # 转移矩阵、发射矩阵
    transaction_matrix = np.zeros((len(tags_list), len(tags_list)), dtype=float)
    emission_matrix = np.zeros((len(tags_list), len(observation)), dtype=float)

    # 计算转移矩阵和发射矩阵
    word_file = open('../../../data/chapter3/hmm_pos/199801.txt','r',encoding='utf-8').readlines()
    for line in word_file:
        if line.strip() != '':
            word_pos_list = line.strip().split('  ')
            for i in range(1, len(word_pos_list)):
                tag = word_pos_list[i].split('/')[1]
                pre_tag = word_pos_list[i - 1].split('/')[1]
                try:
                    transaction_matrix[tags_list.index(pre_tag)][tags_list.index(tag)] += 1
                    tags_num[tag] += 1
                except ValueError:
                    if ']' in tag:
                        tag = tag.split(']')[0]
                    else:
                        pre_tag = tag.split(']')[0]
                    transaction_matrix[tags_list.index(pre_tag)][tags_list.index(tag)] += 1
                    tags_num[tag] += 1

            for o in observation:
                # 注意' 我/'，'我/'的区别
                if ' ' + o in line:
                    pos_tag = line.strip().split(o)[1].split('  ')[0].strip('/')
                    if ']' in pos_tag:
                        pos_tag = pos_tag.split(']')[0]
                    emission_matrix[tags_list.index(pos_tag)][observation.index(o)] += 1

    # # 加一法平滑方法
    # for row in range(transaction_matrix.shape[0]):
    # 	transaction_matrix[row] += 1
    # 	transaction_matrix[row] /= np.sum(transaction_matrix[row])
    # for row in range(emission_matrix.shape[0]):
    #     emission_matrix[row] += 1
    #     emission_matrix[row] /= tags_num[tags_list[row]] + emission_matrix.shape[1]
    #
    # # Laplace平滑方法 l=1，p=1e-16
    # # 这里也可以看做使用一个极小的数字如1e-16代替0，分母+1为避免分母为0的情况出现
    # for row in range(transaction_matrix.shape[0]):
    # 	n = np.sum(transaction_matrix[row])
    # 	transaction_matrix[row] += 1e-16
    # 	transaction_matrix[row] /= n + 1
    # for row in range(emission_matrix.shape[0]):
    #     emission_matrix[row] += 1e-16
    #     emission_matrix[row] /= tags_num[tags_list[row]] + 1

    # Laplace平滑方法 l=1，p=1e-16
    # 这里也可以看做使用一个极小的数字如1e-16代替0，分母+1为避免分母为0的情况出现
    for row in range(transaction_matrix.shape[0]):
        n = np.sum(transaction_matrix[row])
        transaction_matrix[row] += 1e-16
        transaction_matrix[row] /= n + 1

    for row in range(emission_matrix.shape[0]):
        emission_matrix[row] += 1e-16
        emission_matrix[row] /= tags_num[tags_list[row]] + 1

    times_sum = sum(tags_num.values())
    for item in tags_num.keys():
        tags_num[item] = tags_num[item] / times_sum

    # 返回隐状态，初始概率，转移概率，发射矩阵概率
    return tags_list, list(tags_num.values()), transaction_matrix, emission_matrix


if __name__ == '__main__':

    input_str = "今天和明天我弹琴。"
    obs = seg(input_str).strip().split(' ')
    hid, init_p, trans_p, emit_p = cal_hmm_matrix(obs)

    result = viterbi(len(obs), len(hid), init_p, trans_p, emit_p)

    tag_line = ''
    for k in range(len(result)):
        tag_line += obs[k] + hid[int(result[k])] + ' '
    print(tag_line)
