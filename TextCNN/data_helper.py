import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import pandas as pd

def load_data_and_labels(file_dir):
    files = os.listdir(file_dir)
    labels = []
    data = []
    max_sequence_length = 0
    index = 0
    for file in files:
        df = pd.read_csv(os.path.join(file_dir, file), header=None, delimiter=None, encoding="iso-8859-1", error_bad_lines=False)
        for line in list(df[1]):
            if len(line) > max_sequence_length:
                max_sequence_length = len(line)
            data.append(line)
            labels.append(index)
        index += 1

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    print(tokenizer.word_counts)
    sequences = tokenizer.texts_to_sequences(data)
    data = pad_sequences(sequences, max_sequence_length, padding='post')
    return np.array(data), to_categorical(labels), len(tokenizer.word_index)


if __name__ == '__main__':
    x, y, num_words = load_data_and_labels('../data/')
    print(x[0])
    print(np.array(x).shape, np.array(y).shape)
