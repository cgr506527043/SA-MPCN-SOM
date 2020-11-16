# -*- coding: utf-8 -*-
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import numpy.linalg as LA
from collections import defaultdict
from sklearn import metrics

# -----------------------获取中心样本索引------------------------------
def center_index(som_centers):
    index = []
    for center in som_centers:
        distances = LA.norm(np.array(center)-dataset_old, ord=2, axis=1).tolist()
        index.append(distances.index(min(distances)))
    return index

def get_som_center(classfiy):
    som_centers = []
    for i in classify.values():
        som_centers.append((np.sum(dataset_old[i], axis=0) / len(i)).tolist())
    index = center_index(som_centers)
    return index

def loadData(filepath, has_id, class_position):
    with open(filepath) as f:
        lines = (line.strip() for line in f)
        dataset = np.loadtxt(lines, delimiter=',', dtype=np.str, comments="#")
    if has_id:
        # Remove the first column (ID)
        dataset = np.delete(dataset, 0, axis=1)
    if class_position == 'first':
        classes = dataset[:, 0]
        dataset = np.delete(dataset, 0, axis=1)
        dataset = np.asarray(dataset, dtype=np.float)
    else:
        classes = dataset[:, -1]
        dataset = np.delete(dataset, -1, axis=1)
        dataset = np.asarray(dataset, dtype=np.float)
    return dataset, classes

def normal_X(X):
    """
    :param X:二维矩阵，N*D，N个D维的数据
    :return: 将X归一化的结果
    """
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X

def normal_W(W):
    """
    :param W:二维矩阵，D*(n*m)，D个n*m维的数据
    :return: 将W归一化的结果
    """
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:, i], W[:, i]))
        W[:, i] /= np.sqrt(temp)
    return W

# 神经网络
class SOM(object):
    def __init__(self, X, output, iteration, batch_size):
        """
        :param X: 形状是N*D,输入样本有N个,每个D维
        :param output: (n,m)一个元组，为输出层的形状是一个n*m的二维矩阵
        :param iteration:迭代次数
        :param batch_size:每次迭代时的样本数量
        初始化一个权值矩阵，形状为D*(n*m)，即有n*m权值向量，每个D维
        """
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        self.W = np.random.rand(X.shape[1], output[0] * output[1])

    # 获取领域半径
    def GetN(self, t):
        """
        :param t:时间t, 这里用迭代次数来表示时间
        :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
        """
        a = min(self.output)        # a表示与输出结点有关的正常数
        return int(a - float(a) * t / self.iteration)

    # 求学习率
    def Geteta(self, t, n):
        """
        :param t: 时间t, 这里用迭代次数来表示时间
        :param n: 拓扑距离
        :return: 返回学习率，
        """
        return np.power(np.e, -n) / (t + 2)

    def getneighbor(self, index, N):
        """
        :param index:获胜神经元的下标
        :param N: 邻域半径
        :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
        """
        a, b = self.output
        length = a * b      # 采用矩形邻域

        def distence(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b  # //:向下取整; %:返回除法的余数;
            i2_a, i2_b = index2 // a, index2 % b
            return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)  # abs() 函数返回数字的绝对值。

        ans = [set() for i in range(N + 1)]
        for i in range(length):
            dist_a, dist_b = distence(i, index)
            if dist_a <= N and dist_b <= N:
                ans[max(dist_a, dist_b)].add(i)     # 切比雪夫距离
        return ans

    # 更新权值矩阵
    def updata_W(self, X, t, winner):
        N = self.GetN(t)  # 表示随时间变化的拓扑距离
        for x, i in enumerate(winner):
            # 邻域需要更新的神经元
            to_update = self.getneighbor(i[0], N)
            for j in range(N + 1):
                e = self.Geteta(t, j)  # 表示学习率
                for w in to_update[j]:
                    self.W[:, w] = np.add(self.W[:, w], e * (X[x, :] - self.W[:, w]))       # 更新连接权值

    def train(self):
        """
        train_Y:训练样本与形状为batch_size*(n*m)
        winner:一个一维向量，batch_size个获胜神经元的下标
        :return:返回值是调整后的W
        """
        count = 0
        while self.iteration > count:
            print("第%d轮迭代" % count)
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
            self.W = normal_W(self.W)
            train_X = normal_X(train_X)
            train_Y = train_X.dot(self.W)
            winner = np.argmax(train_Y, axis=1).tolist()        # 获胜结点下标
            print(winner)
            # squeeze_winner = np.squeeze(winner).tolist()
            # print("winner length:", len(set(squeeze_winner)))

            res_result = np.squeeze(self.train_result()).tolist()
            classify = defaultdict(list)
            for k, va in [(v, i) for i, v in enumerate(res_result)]:
                classify[k].append(va)
            classify = dict(classify)

            # accuracy
            acc = 0
            for i in classify.values():
                acc += Counter(np.array(classes)[i]).most_common(1)[0][1]
            print("准确率：%.10f" % (acc / len(res_result)))

            self.updata_W(train_X, count, winner)               # 更新获胜结点及周边结点权值
            count += 1
        return self.W

    # 输出获胜神经元结果
    def train_result(self):
        normal_X(self.X)
        train_Y = self.X.dot(self.W)
        winner = np.argmax(train_Y, axis=1).tolist()
        return winner

# 画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm','#B8860B','#C5C1AA','#CD00CD','#FF7F00']
    # style.use('ggplot')
    fig = pl.figure()
    ax = Axes3D(fig)
    for i in range(len(C)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        coo_Z = []  # z坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
            coo_Z.append(C[i][j][2])
        ax.scatter(coo_X, coo_Y, coo_Z, marker='x', color=colValue[i % len(colValue)], label=i)
    ax.legend(loc='upper right')  # 图例位置
    pl.show()

# dataset, classes = loadData(filepath="BERT/ATT_DPTC.txt", has_id=None, class_position='last')
dataset = np.load("../BERT/embeddings.npy")
# print(dataset)
classes = [np.nonzero(i)[0][0] for i in np.load("../BERT/labels.npy").tolist()]
# print(classes)

dataset_old = dataset.copy()
classes_old = classes.copy()
dataset = np.mat(dataset)

som = SOM(dataset, (1, len(set(classes))), 11, len(dataset))
layers = som.train()
res = som.train_result()
res_result = np.squeeze(res).tolist()


classify = defaultdict(list)
for k, va in [(v, i) for i, v in enumerate(res_result)]:
    classify[k].append(va)
classify = dict(classify)
print(classify)

# accuracy
acc = 0
for i in classify.values():
    acc += Counter(np.array(classes)[i]).most_common(1)[0][1]
print("准确率：%.10f" % (acc / len(res_result)))



# update result
# for i, j in classify.items():
#     print(np.array(res_result)[j])
#     print(Counter(np.array(classes)[j]).most_common(1)[0][0])
#     for k in range(len(np.array(res_result)[j])):
#         np.array(res_result)[j][k] = Counter(np.array(classes)[j]).most_common(1)[0][0]


# Homogeneity
## h, c, v = metrics.homogeneity_completeness_v_measure(labels_true=classes_old, labels_pred=res_result)
# print("Homogeneity: %0.3f" % h)
# print("Completeness: %0.3f" % c)
# print("V-measure: %0.3f" % v)

# RI
# nmi = metrics.normalized_mutual_info_score(classes_old, res_result)
# rand = metrics.adjusted_rand_score(classes_old, res_result)
# print("NMI: %0.3f" % nmi)
# print("Rand score: %0.3f" % rand)

# Silhouette Coefficient
sc = metrics.silhouette_score(dataset, res_result)
print("SC: %0.3f" % sc)

# Calinski Harabasz
ch = metrics.calinski_harabaz_score(dataset, res_result)
print("CH: %0.3f" % ch)

# DBI
dbi = metrics.davies_bouldin_score(dataset, res_result)
print("DBI: %0.3F" % dbi)
