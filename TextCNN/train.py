from __future__ import print_function
import numpy as np
from data_helper import load_data_and_labels
from model import TextCNN
from keras import backend as K
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CORPUS_DIR = '../data/'

EPOCHS = 10
BATCH_SIZE = 32
EMBEDDING_SIZE = 256
NUM_FILTERS = 128
FILTER_SIZES = [3, 4, 5]

# Load data and labels
data, labels, num_words = load_data_and_labels(CORPUS_DIR)
# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# x_train = data
# y_train = labels

VALIDATION_SPLIT = 0.1
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

start_time = time.time()
print('Training model.')
text_cnn = TextCNN(num_class=y_train.shape[1],
                   num_words=num_words,
                   sequence_length=data.shape[1],
                   embedding_size=EMBEDDING_SIZE,
                   num_filters=NUM_FILTERS,
                   filter_sizes=FILTER_SIZES)

model = text_cnn.model
# model.summary()
# plot_model(model=model, to_file="model.png", show_shapes=True)
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# flatten_layer = K.function([model.get_layer("input").input, K.learning_phase()], [model.get_layer("flatten").output])
# flatten_layer_vec = flatten_layer([x_train, 0])[0]
# print(flatten_layer_vec)


# with open("soft.txt", "w", encoding="utf-8") as f:
#     for i, j in enumerate(flatten_layer_vec):
#         for k in j:
#             f.write(str(k)+",")
#         f.write(str(list(np.nonzero(y_train[i]))[0][0]))
#         f.write("\n")


score = model.evaluate(x_train, y_train, verbose=0)
print('train score:', score[0])
print('train accuracy:', score[1])

score = model.evaluate(x_val,  y_val, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
# model.save("model.h5")


