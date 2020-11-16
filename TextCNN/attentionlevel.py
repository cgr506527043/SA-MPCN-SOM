from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer
class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)

class Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim=output_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[-1],self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[-1],self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[-1], self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x):
        Q_seq, K_seq, V_seq = x,x,x
        Q_seq = K.dot(Q_seq, self.WQ)
        print(K.int_shape(x))
        print(K.int_shape(self.WQ))
        K_seq = K.dot(K_seq, self.WK)
        V_seq = K.dot(V_seq, self.WV)
        A = K.batch_dot( Q_seq,K_seq,axes=[2, 2]) / K.int_shape(x)[-1]** 0.5
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[2, 1])
        print(K.int_shape(O_seq))
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)