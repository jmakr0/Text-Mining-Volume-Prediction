from random import sample, seed

import numpy as np
from keras.callbacks import CSVLogger
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, concatenate
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Input, Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from src.models.prediction_data import PredictionData
from src.utils.settings import Settings


class HeadlinePrediction:

    MAX_FEATURES = 40000
    MAX_LENGTH = 20
    META_EMBEDDING_DIMS = 64
    BATCH_SIZE = 32
    EMBEDDING_DIMS = 50
    EPOCHS = 20

    def __init__(self):
        self.data = PredictionData()
        self.data.prepare()
        self.data.test_print()

        self.word_tokenizer = Tokenizer(self.MAX_FEATURES)
        self.word_tokenizer.fit_on_texts(self.data.get_headlines())
        self._test_print_word_tokenizer()

        self.titles_tf = self.word_tokenizer.texts_to_sequences(self.data.get_headlines())
        self.titles_tf = sequence.pad_sequences(self.titles_tf, maxlen=self.MAX_LENGTH)

        self.embedding_vectors = {}
        self._add_embeddings()

        self.weights_matrix = np.zeros((self.MAX_FEATURES + 1, 50))
        self._initialize_weigts()

        # Text Branch

        self.titles_input = Input(shape=(self.MAX_LENGTH,), name='guardian_headlines_input')
        self.titles_embedding = Embedding(self.MAX_FEATURES + 1, self.EMBEDDING_DIMS, weights=[self.weights_matrix])(self.titles_input)
        self.titles_pooling = GlobalAveragePooling1D()(self.titles_embedding)

        self.aux_output = Dense(1, activation='sigmoid', name='aux_out')(self.titles_pooling)

        # Metadata Branch

        self.hours_input = Input(shape=(1,), name='hours_input')
        self.hours_embedding = Embedding(24, self.META_EMBEDDING_DIMS)(self.hours_input)
        self.hours_reshape = Reshape((self.META_EMBEDDING_DIMS,))(self.hours_embedding)

        self.dayofweeks_input = Input(shape=(1,), name='dayofweeks_input')
        self.dayofweeks_embedding = Embedding(7, self.META_EMBEDDING_DIMS)(self.dayofweeks_input)
        self.dayofweeks_reshape = Reshape((self.META_EMBEDDING_DIMS,))(self.dayofweeks_embedding)

        self.minutes_input = Input(shape=(1,), name='minutes_input')
        self.minutes_embedding = Embedding(60, self.META_EMBEDDING_DIMS)(self.minutes_input)
        self.minutes_reshape = Reshape((self.META_EMBEDDING_DIMS,))(self.minutes_embedding)

        self.dayofyears_input = Input(shape=(1,), name='dayofyears_input')
        self.dayofyears_embedding = Embedding(366, self.META_EMBEDDING_DIMS)(self.dayofyears_input)
        self.dayofyears_reshape = Reshape((self.META_EMBEDDING_DIMS,))(self.dayofyears_embedding)

        # Merge the Branches and Complete Model

        self.merged = concatenate([self.titles_pooling,
                                   self.hours_reshape,
                                   self.dayofweeks_reshape,
                                   self.minutes_reshape,
                                   self.dayofyears_reshape])

        self.hidden_1 = Dense(256, activation='relu')(self.merged)
        self.hidden_1 = BatchNormalization()(self.hidden_1)

        self.main_output = Dense(1, activation='sigmoid', name='main_out')(self.hidden_1)

        # compile module

        self.model = Model(inputs=[self.titles_input,
                                   self.hours_input,
                                   self.dayofweeks_input,
                                   self.minutes_input,
                                   self.dayofyears_input], outputs=[self.main_output, self.aux_output])

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'],
                      loss_weights=[1, 0.2])

        self.model.summary()

    def _add_embeddings(self):
        settings = Settings()
        embeddings_path = settings.get_glove_embedding()

        with open(embeddings_path, 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                vec = np.array(line_split[1:], dtype=float)
                word = line_split[0]
                self.embedding_vectors[word] = vec

        print(self.embedding_vectors['you'])

    def _initialize_weigts(self):
        for word, i in self.word_tokenizer.word_index.items():

            embedding_vector = self.embedding_vectors.get(word)
            if embedding_vector is not None and i <= self.MAX_FEATURES:
                self.weights_matrix[i] = embedding_vector

        # index 0 vector should be all zeroes, index 1 vector should be the same than self.embedding_vectors['you']
        print(self.weights_matrix[0:2, :])

    def _test_print_word_tokenizer(self):
        print(str(self.word_tokenizer.word_counts)[0:100])
        print(str(self.word_tokenizer.word_index)[0:100])
        print(len(self.word_tokenizer.word_counts))

    def train(self):
        seed(123)
        split = 0.2

        # returns randomized indices with no repeats
        idx = sample(range(self.titles_tf.shape[0]), self.titles_tf.shape[0])

        titles_tf = self.titles_tf[idx, :]
        hours = self.data.get_hours()[idx]
        dayofweeks = self.data.get_dayofweeks()[idx]
        minutes = self.data.get_minutes()[idx]
        dayofyears_tf = self.data.get_dayofyears()[idx]
        is_top_submission = self.data.get_is_top_submission()[idx]

        print(1 - np.mean(is_top_submission[:(int(titles_tf.shape[0] * split))]))
        csv_logger = CSVLogger('training.csv')

        self.model.fit([titles_tf, hours, dayofweeks, minutes, dayofyears_tf], [is_top_submission, is_top_submission],
                  batch_size=self.BATCH_SIZE,
                  epochs=self.EPOCHS,
                  validation_split=split, callbacks=[csv_logger])
