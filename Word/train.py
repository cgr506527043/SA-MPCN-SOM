from __future__ import print_function
import os
import numpy as np
from data_helper import load_data_and_labels
from model import TextCNN
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from keras import backend as K

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# parameter
CORPUS_DIR = '../data'
EPOCHS = 1
BATCH_SIZE = 64
EMBEDDING_SIZE = 300
NUM_FILTERS = 64
FILTER_SIZES = [3, 4, 5]

# Load data and labels
# data, labels, num_words, embeddings = load_data_and_labels(CORPUS_DIR)
data = np.load("npy/data.npy")
labels = np.load("npy/labels.npy")
embeddings = np.load("npy/embeddings.npy")
num_words = len(np.load("npy/embeddings.npy"))

# split the data into a training set and a validation set
# x_train, x_val, y_train, y_val = train_test_split(data, labels, train_size=0.9, random_state=23, shuffle=True)

print('Training model.')
text_cnn = TextCNN(num_class=labels.shape[1],
                   num_words=num_words,
                   sequence_length=data.shape[1],
                   embedding_size=EMBEDDING_SIZE,
                   num_filters=NUM_FILTERS,
                   filter_sizes=FILTER_SIZES,
                   embedding_matrix=embeddings)

model = text_cnn.model
# model.summary()
# plot_model(model=model, to_file="model.png", show_shapes=True)
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor="val_loss", patience=3, verbose=0, mode="min")

model.fit(data, labels,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_split=0.1,
          callbacks=[earlyStopping],
          verbose=1)

'''
score = model.evaluate(x_train, y_train, verbose=0)
print('train score:', score[0])
print('train accuracy:', score[1])
# model.save("model.h5")

score = model.evaluate(x_val,  y_val, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

'''
flatten_layer = K.function([model.get_layer("input").input, K.learning_phase()], [model.get_layer("flatten").output])
flatten_layer_vec = flatten_layer([data, 0])[0]

with open("soft.txt", "w", encoding="utf-8") as f:
    for i, j in enumerate(flatten_layer_vec):
        for k in j:
            f.write(str(k)+",")
        f.write(str(list(np.nonzero(labels[i]))[0][0]))
        f.write("\n")
