import numpy as np

from src.utils.settings import Settings


class Glove:
    DIMENSIONS = 50

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        settings = Settings()
        self.embedding_path = settings.get_glove_embedding()

        self.weights_matrix = np.zeros((self.tokenizer.num_words, self.DIMENSIONS))

    def load_embedding(self):
        # This is a helper dict that temporary stores the contents of the glove file
        glove_word_vectors = {}
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_split = line.strip().split()
                word = line_split[0]
                vector = np.array(line_split[1:], dtype=float)
                glove_word_vectors[word] = vector

        for word, index in self.tokenizer.word_index.items():
            word_vector = glove_word_vectors.get(word)
            if word_vector is not None and index < self.tokenizer.num_words:
                self.weights_matrix[index] = word_vector
