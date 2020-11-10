import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from self_attention import Attention
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras import regularizers, callbacks

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_data_and_labels(file_dir):
    EMBEDDING_FILE = '../Word/GoogleNews-vectors-negative300.bin'
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

class DPCNN:
    def __init__(self, num_class,
                 num_words,
                 sequence_length,
                 embedding_size,
                 num_filters,
                 filter_sizes,
                 embedding_matrix):

        filter_nr = num_filters
        filter_size = filter_sizes
        max_pool_size = 3
        max_pool_strides = 2
        dense_nr = 256
        spatial_dropout = 0.2
        dense_dropout = 0.5
        train_embed = False
        conv_kern_reg = regularizers.l2(0.00001)
        conv_bias_reg = regularizers.l2(0.00001)

        comment = Input(shape=(sequence_length,), name="input")
        emb_comment = Embedding(num_words, embedding_size, weights=[embedding_matrix], trainable=train_embed)(comment)
        # emb_comment = Attention(int(emb_comment.shape[-1]))(emb_comment)
        emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)
        block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
        block1 = BatchNormalization()(block1)
        block1 = PReLU()(block1)

        # we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
        # if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
        resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='relu',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
        resize_emb = PReLU()(resize_emb)

        block1_output = add([block1, resize_emb])
        block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

        block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
        block2 = BatchNormalization()(block2)
        block2 = PReLU()(block2)
        block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
        block2 = BatchNormalization()(block2)
        block2 = PReLU()(block2)

        block2_output = add([block2, block1_output])
        block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

        block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
        block3 = BatchNormalization()(block3)
        block3 = PReLU()(block3)
        block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
        block3 = BatchNormalization()(block3)
        block3 = PReLU()(block3)

        block3_output = add([block3, block2_output])
        block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

        block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
        block4 = BatchNormalization()(block4)
        block4 = PReLU()(block4)
        block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
        block4 = BatchNormalization()(block4)
        block4 = PReLU()(block4)

        block4_output = add([block4, block3_output])
        block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

        block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
        block5 = BatchNormalization()(block5)
        block5 = PReLU()(block5)
        block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='relu',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
        block5 = BatchNormalization()(block5)
        block5 = PReLU()(block5)

        output = Add(name="add")([block5, block4_output])

        output = Dense(dense_nr, activation='relu')(output)
        output = BatchNormalization()(output)
        output = PReLU()(output)
        output = Dropout(dense_dropout)(output)
        output = Flatten()(output)
        output = Dense(num_class, activation='relu')(output)

        self.model = Model(comment, output)
        self.model.trainable = False
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

def schedule(ind):
    a = [0.0001, 0.0005, 0.0001, 0.0001]
    return a[ind]

epochs = 4
batch_size = 128
EMBEDDING_SIZE = 768
NUM_FILTERS = 64
FILTER_SIZES = 3

CORPUS_DIR = '../data/'
# data, labels, num_words, embeddings = load_data_and_labels(CORPUS_DIR)
data = np.load("../BERT/data.npy")
labels = np.load("../BERT/labels.npy")
embeddings = np.load("../BERT/embeddings.npy")
num_words = len(np.load("../BERT/embeddings.npy"))

model = DPCNN(num_class=labels.shape[1],
              num_words=num_words,
              sequence_length=data.shape[1],
              embedding_size=EMBEDDING_SIZE,
              num_filters=NUM_FILTERS,
              filter_sizes=FILTER_SIZES,
              embedding_matrix=embeddings).model

Xtrain, Xval, ytrain, yval = train_test_split(data, labels, train_size=0.9, random_state=233)
lr = callbacks.LearningRateScheduler(schedule)
ra_val = RocAucEvaluation(validation_data=(Xval, yval), interval=1)

print("train model.")
model.fit(data, labels, batch_size=batch_size, epochs=epochs, callbacks=[lr], verbose=1)
# model.fit(Xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_data=(Xval, yval), callbacks=[lr, ra_val], verbose=1)
# score = model.evaluate(Xval,  yval, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


flatten_layer = K.function([model.get_layer("input").input, K.learning_phase()], [model.get_layer("add").output])
flatten_layer_vec = np.squeeze(flatten_layer([data, 0])[0])
print(flatten_layer_vec)
print(flatten_layer_vec[0])
with open("soft.txt", "w", encoding="utf-8") as f:
    for i, j in enumerate(flatten_layer_vec):
        for k in j:
            f.write(str(k)+",")
        f.write(str(list(np.nonzero(labels[i]))[0][0]))
        f.write("\n")


