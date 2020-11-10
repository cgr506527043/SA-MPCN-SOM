# coding=utf-8
from __future__ import print_function
from text_birnn import TextBiRNN
import numpy as np
import os
from keras import backend as K
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CORPUS_DIR = '../data'

epochs = 1
EMBEDDING_SIZE = 768
batch_size = 64
CORPUS_DIR = '../data/'

print('Loading data...')
# Load data and labels
data = np.load("../npy/data.npy")
labels = np.load("../npy/labels.npy")
embeddings = np.load("../npy/embeddings.npy")
num_words = len(np.load("../npy/embeddings.npy"))

print('Build model...')
model = TextBiRNN(embedding_matrix=embeddings, maxlen=data.shape[1], max_features=num_words,
                embedding_dims=EMBEDDING_SIZE, class_num=labels.shape[1]).get_model()

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=1)

flatten_layer = K.function([model.get_layer("input").input, K.learning_phase()], [model.get_layer("bilstm").output])
flatten_layer_vec = flatten_layer([data, 0])[0]
print(flatten_layer_vec)
with open("birnn.txt", "w", encoding="utf-8") as f:
    for i, j in enumerate(flatten_layer_vec):
        for k in j:
            f.write(str(k)+",")
        f.write(str(list(np.nonzero(labels[i]))[0][0]))
        f.write("\n")