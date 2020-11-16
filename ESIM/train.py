from __future__ import print_function
import os
import numpy as np
from data_helper import Dataloader
from models import get_ESIM_model
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from keras import backend as K

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parameter
CORPUS_DIR = '../data'
RECURRENT_UNITS = 30
BATCH_SIZE = 128
EMBEDDING_SIZE = 300
DROPOUT_RATE = 0.5
DENSE_UNITS = 300
EPOCHS = 1

# Load data and labels
data, labels, num_words, embeddings = Dataloader(CORPUS_DIR)
# data = np.load("../BERT/data.npy")
# labels = np.load("../BERT/labels.npy")
# embeddings = np.load("../BERT/embeddings.npy")
# num_words = len(np.load("../BERT/embeddings.npy"))

x_train, x_val, y_train, y_val = train_test_split(data, labels, train_size=0.9, random_state=23, shuffle=True)

print('Training model.')
esim=get_ESIM_model(nb_words=num_words, embedding_dim=EMBEDDING_SIZE, embedding_matrix=embeddings,
                    recurrent_units=RECURRENT_UNITS, dense_units=DENSE_UNITS, dropout_rate=DROPOUT_RATE,
                    max_sequence_length=data.shape[1], out_size=labels.shape[1])


earlyStopping = EarlyStopping(monitor="val_loss", patience=5, verbose=0, mode="min")
esim.fit([data, data], labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=[earlyStopping], verbose=1)

# score = esim.evaluate([x_train, x_train], y_train, verbose=0)
# print('train score:', score[0])
# print('train accuracy:', score[1])
# model.save("model.h5")

# score = esim.evaluate([x_val, x_val],  y_val, verbose=0)
#
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

input1 = esim.get_layer("q1").input
input2 = esim.get_layer("q2").input
output = esim.get_layer("final").output
model_penultimate = Model([input1, input2], output)

# inference of penultimate layer
H = np.squeeze(model_penultimate.predict([data, data]))
print("Sample shape: {}".format(H.shape))

with open("soft.txt", "w", encoding="utf-8") as f:
    for i, j in enumerate(H):
        for k in j:
            f.write(str(k)+",")
        f.write(str(np.nonzero(y_train[i])[0][0])+"\n")
