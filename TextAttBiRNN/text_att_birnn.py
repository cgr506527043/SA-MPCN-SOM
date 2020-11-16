# coding=utf-8

from keras import Input, Model
from keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM, Flatten

from attention import Attention


class TextAttBiRNN(object):
    def __init__(self, embedding_matrix, maxlen, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding_matrix = embedding_matrix
    def get_model(self):
        input = Input((self.maxlen,), name="input")

        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen, weights=[self.embedding_matrix])(input)
        x = Bidirectional(LSTM(128, return_sequences=True))(embedding)  # LSTM or GRU
        x = Attention(self.maxlen, name="attention")(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        model.trainable = False
        return model
