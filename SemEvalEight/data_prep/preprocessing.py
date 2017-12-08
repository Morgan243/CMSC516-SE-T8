# Author: Morgan Stuart

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def tokenize_texts(texts, nb_words=1000,
                   lower=True, char_level=False):
    tokenizer = Tokenizer(num_words=nb_words, lower=lower,
                          char_level=char_level)
    tokenizer.fit_on_texts(texts=texts)
    sequences = tokenizer.texts_to_sequences(texts=texts)

    return tokenizer, sequences
