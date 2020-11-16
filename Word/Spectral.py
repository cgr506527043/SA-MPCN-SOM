import numpy as np
from sklearn.cluster import SpectralClustering
from collections import defaultdict, Counter

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

dataset, classes = loadData(filepath="./BERT/ATT_DPTC.txt", has_id=None, class_position='last')

spectral = SpectralClustering(n_clusters=len(set(classes)), affinity="nearest_neighbors", n_neighbors=10, gamma=2.0)
pred_y = spectral.fit_predict(dataset)
print(pred_y)

classify = defaultdict(list)
for k, va in [(v, i) for i, v in enumerate(pred_y)]:
    classify[k].append(va)
classify = dict(classify)
print(classify)

# accuracy
acc = 0
for i in classify.values():
    acc += Counter(np.array(classes)[i]).most_common(1)[0][1]
print("准确率：%.10f" % (acc / len(pred_y)))