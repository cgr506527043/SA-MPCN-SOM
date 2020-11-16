from keras.models import Model
from keras.layers import *
from self_attention import Attention

class TextCNN:
    def __init__(self, num_class,
                 num_words,
                 sequence_length,
                 embedding_size,
                 num_filters,
                 filter_sizes,
                 embedding_matrix):

        # input layer
        sequence_input = Input(shape=(sequence_length,), dtype='int32', name="input")
        # embedding layer
        embedding_layer = Embedding(
            input_dim=num_words,
            output_dim=embedding_size,
            weights=[embedding_matrix],
            input_length=sequence_length,
            trainable=False
        )

        embedded_sequences = embedding_layer(sequence_input)
        # embedded_sequences = Attention(int(embedded_sequences.shape[-1]))(embedded_sequences)

        # create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            x = Conv1D(num_filters, filter_size, activation='relu', trainable=False)(embedded_sequences)
            # x = Conv1D(num_filters, int(x.shape[1]), activation='relu')(x)
            x = GlobalMaxPooling1D(trainable=False)(x)
            pooled_outputs.append(x)
        merged = Concatenate(name="flatten", trainable=False)(pooled_outputs)
        outputs = Dense(num_class, activation='softmax', trainable=False)(merged)
        self.model = Model(sequence_input, outputs)
        self.model.trainable = False
