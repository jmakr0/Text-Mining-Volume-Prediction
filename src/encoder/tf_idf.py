import numpy as np
from keras.preprocessing.text import Tokenizer

from src.encoder.glove import Glove
from src.utils.settings import Settings


class TfIdf:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_tfidf_matrix_for_corpus(self, corpus):
        return self.tokenizer.texts_to_matrix(corpus, mode='tfidf')



