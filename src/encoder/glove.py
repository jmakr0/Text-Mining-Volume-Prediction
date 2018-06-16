import numpy as np

from src.utils.settings import Settings


class Glove:
    DIMENSIONS = 50

    def __init__(self, dictionary_size):
        self.dictionary_size = dictionary_size
        self.dimensions = self.DIMENSIONS

        settings = Settings()
        self.embedding_path = settings.get_glove_embedding()

        self.word_numbers = {}
        self.embedding_vectors = np.zeros((dictionary_size + 1, self.DIMENSIONS))

    def load_embedding(self):
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line_split = line.strip().split()
                word = line_split[0]
                vector = np.array(line_split[1:], dtype=float)

                self.word_numbers[word] = line_number
                self.embedding_vectors[line_number] = vector

                if line_number == self.dictionary_size:
                    break

    def get_word_number(self, word):
        if len(self.word_numbers) == 0:
            raise Exception('No embedding loaded.')

        word = word.lower()
        if word in self.word_numbers:
            return self.word_numbers[word]
        return 0

    def get_word_vector(self, word):
        return self.embedding_vectors[self.get_word_number(word)]

    def _text_to_word_sequence(self, text):
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = dict((c, ' ') for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)

        seq = text.split()
        return [word for word in seq if word]

    def text_to_sequence(self, text):
        vector = []

        if isinstance(text, list):
            seq = text
        else:
            seq = self._text_to_word_sequence(text)

        for word in seq:
            vector.append(self.get_word_number(word))

        return vector

    def texts_to_sequences(self, texts):
        result = []

        for text in texts:
            result.append(self.text_to_sequence(text))
        return result
