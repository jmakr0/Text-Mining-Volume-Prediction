import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from src.encoder.glove import Glove
from src.encoder.tf_idf import TfIdf


class GloveTfIdf:
    # TODO parameter max words
    def __init__(self, corpus, max_words=40000):
        self.corpus = corpus
        self.max_words = max_words

        self.tokenizer = Tokenizer(self.max_words)
        self.tokenizer.fit_on_texts(self.corpus)

        self.glove = Glove(self.tokenizer, self.max_words)

        self.tf_idf = TfIdf(self.tokenizer)

        # will contain weighted corpus embeddings using glove and tfidf-score
        self.corpus_glove_tf_idf = []

    def create_weighted_glove(self):
        tf_idf_matrix = self.tf_idf.get_tfidf_matrix_for_corpus(self.corpus)

        for i, corpus_elem in enumerate(self.corpus):
            # the corresponding tfidf-scores generate by the tokenizer for one specific corpus_elem
            tf_idf_vector = tf_idf_matrix[i]

            # will contain the weighted embeddings for all words of the given corpus
            weighted_corpus_embedding = []

            normalized_corpus_elem = text_to_word_sequence(corpus_elem)
            for word in normalized_corpus_elem:
                # the index of the given word as defined by the tokenizer
                word_index = self.tokenizer.word_index[word]

                # the glove embedding for the given word
                word_embedding = self.glove.get_word_index_embedding(word_index)

                # tfidf-scores generate by the tokenizer for one specific corpus_elem
                word_tfidf = tf_idf_vector[word_index]

                # apply the tfidf-score to the given word embedding
                scaled_embedding = np.multiply(word_embedding, word_tfidf)

                weighted_corpus_embedding.append(scaled_embedding)

            self.corpus_glove_tf_idf.append(weighted_corpus_embedding)

    def _normalize_word_sequence(self, text):
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = dict((c, ' ') for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)

        seq = text.split()
        return [word.lower() for word in seq if word]
