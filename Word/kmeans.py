from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import numpy.linalg as LA

def loadData(filepath, has_id, class_position):
    with open(filepath, encoding="latin1") as f:
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
    return dataset.tolist(), classes.tolist()

dataset, classes = loadData(filepath="BERT/ATT_DPTC.txt", has_id=False, class_position="last")
clf = KMeans(n_clusters=len(set(classes)), n_init=20)
y_pred = clf.fit_predict(dataset).tolist()
print(y_pred)
print(dict(Counter(y_pred)))

from collections import defaultdict

d = defaultdict(list)
for k, va in [(v, i) for i, v in enumerate(y_pred)]:
    d[k].append(va)
d = dict(d)
print(d)

acc = 0
for i in np.arange(len(set(classes))):
    acc += Counter(np.array(classes)[d[i]]).most_common(1)[0][1]
print(acc / len(dataset))