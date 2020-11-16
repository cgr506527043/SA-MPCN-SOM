from self_attention import Attention
from keras.layers import *
from keras.models import Model
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from keras import optimizers
from keras.utils.np_utils import to_categorical
import os
import pandas as pd
from keras import callbacks
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_data_and_labels(file_dir):
    EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'
    files = os.listdir(file_dir)
    labels = []
    data = []
    index = 0
    for file in sorted(files):
        df = pd.read_csv(os.path.join(file_dir, file),
                         header=None,
                         delimiter=None,
                         encoding="iso-8859-1",
                         error_bad_lines=False)

        for line in list(df[1]):
            data.append(line)
            labels.append(index)
        index += 1
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)

    seq_lens = [len(s) for s in sequences]
    max_sequence_length = max(seq_lens)

    data = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    EMBEDDING_DIM = 300
    nb_words = len(tokenizer.word_index)
    print("word num:", nb_words)

    embedding_matrix = np.zeros((nb_words+1, EMBEDDING_DIM))

    for word, i in tokenizer.word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
        else:
            print(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    return np.array(data), to_categorical(labels), nb_words, embedding_matrix

class ATT_DPTC:
    def __init__(self, num_class,
                 num_words,
                 sequence_length,
                 embedding_size,
                 num_filters,
                 filter_sizes,
                 embedding_matrix):
        dense_nr = 256
        dense_dropout = 0.5
        spatial_dropout = 0.2
        train_embed = False
        conv_kern_reg = regularizers.l2(0.00001)
        conv_bias_reg = regularizers.l2(0.00001)

        sequence_input = Input(shape=(sequence_length,), dtype='int32', name="input")
        emb_comment = Embedding(num_words, embedding_size, weights=[embedding_matrix],
                                trainable=train_embed)(sequence_input)
        emb_comment = Attention(int(emb_comment.shape[-1]))(emb_comment)
        emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

        pooled_outputs = []
        for filter_size in filter_sizes:
            block1 = Conv1D(num_filters, kernel_size=filter_size, padding='same', activation="sigmoid",
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
            block2 = Conv1D(num_filters, kernel_size=filter_size, padding='same', activation="sigmoid", strides=2,
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
            block3 = Conv1D(num_filters, kernel_size=filter_size, padding='same', activation="sigmoid", strides=2,
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
            block4 = Conv1D(num_filters, kernel_size=filter_size, padding='same', activation="sigmoid", strides=2,
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
            block5 = Conv1D(num_filters, kernel_size=filter_size, padding='same', activation="sigmoid", strides=2,
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
            block6 = Conv1D(num_filters, kernel_size=filter_size, padding='same', activation="sigmoid", strides=2,
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
            block5 = Attention(int(block5.shape[-1]))(block5)
            output = GlobalMaxPooling1D()(block5)
            pooled_outputs.append(output)
        merged = Concatenate(name="concatenate")(pooled_outputs)

        output = Dense(num_class, activation='sigmoid')(merged)
        self.model = Model(sequence_input, output)
        self.model.trainable = False
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.Adam(),
                           metrics=['accuracy'])
def schedule(ind):
    a = [0.001, 0.0005, 0.0001, 0.0001]
    return a[ind]

epochs = 4
EMBEDDING_SIZE = 768
NUM_FILTERS = 64
FILTER_SIZES = [3, 4, 5]
batch_size = 64
CORPUS_DIR = '../data/'
# data, labels, num_words, embeddings = load_data_and_labels(CORPUS_DIR)
# batch_size = len(data)

data = np.load("npy/data.npy")
labels = np.load("npy/labels.npy")
embeddings = np.load("npy/embeddings.npy")
num_words = len(np.load("npy/embeddings.npy"))

model = ATT_DPTC(num_class=labels.shape[1],
              num_words=num_words,
              sequence_length=data.shape[1],
              embedding_size=EMBEDDING_SIZE,
              num_filters=NUM_FILTERS,
              filter_sizes=FILTER_SIZES,
              embedding_matrix=embeddings).model
model.summary()
lr = callbacks.LearningRateScheduler(schedule)

print("train model.")
model.fit(data, labels, batch_size=batch_size, epochs=epochs, callbacks=[lr], verbose=1)

# create model that gives penultimate layer
input = model.get_layer("input").input
output = model.get_layer("concatenate").output
model_penultimate = Model(input, output)

# inference of penultimate layer
H = np.squeeze(model_penultimate.predict(data))
print("Sample shape: {}".format(H.shape))

with open("ATT-CONV-DPTC.txt", "w", encoding="utf-8") as f:
    for i, j in enumerate(H):
        for k in j:
            f.write(str(k)+",")
        f.write(str(np.nonzero(labels[i])[0][0]))
        f.write("\n")
