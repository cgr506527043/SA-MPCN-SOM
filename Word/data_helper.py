import os
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd

def load_data_and_labels(file_dir):
    EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'
    files = sorted(os.listdir(file_dir))
    labels = []
    data = []
    index = 0
    for file in files:
        df = pd.read_csv(os.path.join(file_dir, file), header=None, delimiter=None, encoding="iso-8859-1", error_bad_lines=False)
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
    nb_words = len(tokenizer.word_index)+1
    print("word num:", nb_words)

    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

    for word, i in tokenizer.word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
        else:
            print(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    return np.array(data), to_categorical(labels), nb_words, embedding_matrix

if __name__ == '__main__':
    x, y, num_words = load_data_and_labels('../data/')
