 # -*- coding:utf-8 -*-
import random
from numpy import *
import numpy as np
from collections import Counter
# 加载全部数据
def loadDataSet(fileName):
    dataMat = []
    labels = []
    fr = open(fileName, "r", encoding="utf-8")
    for line in fr.readlines():
        curLine = line.strip().split(",")[:-1]
        labels.append(line.strip().split(",")[-1])
        fltLine = list(map(float, curLine))  # transfer to float
        dataMat.append(fltLine)
    return dataMat, labels

# 距离和相似度计算
def pearson_distance(vector1, vector2):
    # scipy.spatial.distance.pdist(X, metric='euclidean', p=2, w=None, V=None, VI=None)
    # 注意，距离转换成相似度时，由于自己和自己的距离是不会计算的默认为0，
    # 所以要先通过dist = spatial.distance.squareform(dist)转换成dense矩阵，再通过1 - dist计算相似度。

    from scipy.spatial.distance import pdist
    X = vstack([vector1, vector2])
    d2 = pdist(X, metric="euclidean")
    return d2


distances_cache = {}

# 总体代价和初始聚类簇
def totalcost(blogwords, costf, medoids_idx):
    size = len(blogwords)
    total_cost = 0.0
    medoids = {}
    for idx in medoids_idx:
        medoids[idx] = []
    for i in range(size):
        choice = None
        min_cost = inf      # inf代表正负无穷
        for m in medoids:
            # 获取字典中m的第i个值
            tmp = distances_cache.get((m, i), None)
            if tmp == None:
                # m和i的距离c
                tmp = pearson_distance(blogwords[m], blogwords[i])
                distances_cache[(m, i)] = tmp
            if tmp < min_cost:
                choice = m
                min_cost = tmp
        medoids[choice].append(i)
        total_cost += min_cost
    return total_cost, medoids


def kmedoids(blogwords, k):
    import random
    size = len(blogwords)

    # 随机抽取k个点作为聚类初始中心

    medoids_idx = random.sample([i for i in range(size)], k)

    # 总代价函数和聚类簇
    pre_cost, medoids = totalcost(blogwords, pearson_distance, medoids_idx)
    print("初始pre_cost:")
    print(pre_cost)
    print("初始medoids:")
    print(medoids)

    # 当前代价
    current_cost = inf  # maxmum of pearson_distances is 2.
    best_choice = []
    best_res = {}
    iter_count = 0
    while 1:
        for m in medoids:
            # medoids的key
            for item in medoids[m]:
                # medoids的value
                if item != m:
                    # mdedoids_idx代表质点列表，idx代表值m在质点列表的索引值
                    idx = medoids_idx.index(m)
                    # mediods的第idx质点放在swap_tmp中
                    swap_temp = medoids_idx[idx]
                    # 更新质点
                    medoids_idx[idx] = item
                    tmp, medoids_ = totalcost(blogwords, pearson_distance, medoids_idx)
                    # print(medoids_)
                    # print tmp,'-------->',medoids_.keys()
                    if tmp < current_cost:
                        best_choice = list(medoids_idx)
                        best_res = dict(medoids_)
                        current_cost = tmp
                    medoids_idx[idx] = swap_temp
        iter_count += 1
        print(current_cost, iter_count)
        if best_choice == medoids_idx:
            break
        if current_cost <= pre_cost:
            pre_cost = current_cost
            medoids = best_res
            medoids_idx = best_choice

    return current_cost, best_choice, best_res

if __name__ == '__main__':
    dataMat, labels = loadDataSet("BERT/ATT_DPTC.txt")
    best_cost, best_choice, best_medoids = kmedoids(dataMat, 11)
    print(best_choice)
    print(best_medoids)
    acc = 0

    for i in best_medoids.values():
        acc += Counter(np.array(labels)[i]).most_common(1)[0][1]
    print("准确率：%.10f" % (acc / len(dataMat)))


