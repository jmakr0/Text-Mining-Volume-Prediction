import numpy as np
from keras.preprocessing.text import Tokenizer

from src.encoder.glove import Glove
from src.encoder.tfidf import TfIdf


class GloveTfIdf:

    def __init__(self, corpus, max_words= 40000):
        self.corpus = corpus
        self.max_words = max_words

        self.tokenizer = Tokenizer(self.max_words)
        self.tokenizer.fit_on_texts(self.corpus)

        self.glove = Glove(self.tokenizer, self.max_words)

        self.tfidf = TfIdf(self.tokenizer)

        # will contain weighted corpus embeddings using glove and tfidf-score
        self.corpus_glove_tfidf = []

    def create_weighted_glove(self):
        tfidf_matrix = self.tfidf.get_tfidf_matrix_for_corpus(self.corpus)

        for i in range(len(self.corpus)):
            corpus_elem = self.corpus[i]

            # the corresponding tfidf-scores generate by the tokenizer for one specific corpus_elem
            tfidf_vector = tfidf_matrix[i]

            # will contain the weighted embeddings for all words of the given corpus
            weighted_corpus_embedding = []

            normalized_corpus_elem = self._normalize_word_sequence(corpus_elem)
            for word in normalized_corpus_elem:
                # the index of the given word as defined by the tokenizer
                word_index = self.tokenizer.word_index[word]

                # the glove embedding for the given word
                word_embedding = self.glove.get_embedding_for_wordindex(word_index)

                # tfidf-scores generate by the tokenizer for one specific corpus_elem
                word_tfidf = tfidf_vector[word_index]

                # apply the tfidf-score to the given word embedding
                scaled_embedding = np.multiply(word_embedding, word_tfidf)

                weighted_corpus_embedding.append(scaled_embedding)

            self.corpus_glove_tfidf.append(weighted_corpus_embedding)

    def _normalize_word_sequence(self, text):
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = dict((c, ' ') for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)

        seq = text.split()
        return [word.lower() for word in seq if word]