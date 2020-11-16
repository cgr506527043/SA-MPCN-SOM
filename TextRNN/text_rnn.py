# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Dropout, LSTM


class TextRNN(object):
    def __init__(self, embedding_matrix, maxlen, max_features, embedding_dims, class_num=1,
                 last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding_matrix = embedding_matrix

    def get_model(self):
        input = Input((self.maxlen,), name="input", dtype="int32")

        embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, weights=[self.embedding_matrix],
                              input_length=self.maxlen, trainable=False)(input)
        x = LSTM(128, name="lstm", trainable=False)(embedding)  # LSTM or GRU
        output = Dense(self.class_num, activation=self.last_activation, trainable=False)(x)
        model = Model(inputs=input, outputs=output)
        model.trainable = False
        return model
