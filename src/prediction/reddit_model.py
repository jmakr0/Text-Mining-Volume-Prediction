import numpy as np

from keras import Input, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Reshape, concatenate, BatchNormalization
from keras.preprocessing import sequence

from src.data_handler.db_fields import LabelsView
from src.prediction.model_builder import ModelBuilder


class RedditModelBuilder(ModelBuilder):
    def __init__(self):
        super().__init__()

        self.required_inputs.append('glove')
        self.required_parameters.append('max_headline_length')

        self.default_parameters['hour_embedding_dimensions'] = 64
        self.default_parameters['minute_embedding_dimensions'] = 64
        self.default_parameters['day_of_week_embedding_dimensions'] = 64
        self.default_parameters['day_of_year_embedding_dimensions'] = 64
        self.default_parameters['relu_fully_connected_dimensions'] = 256
        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'

    def __call__(self):
        super().prepare_building()

        glove = self.inputs['glove']
        headline_input = Input(shape=(self.parameters['max_headline_length'],), name='headline_input')
        headline_embedding = Embedding(glove.embedding_vectors.shape[0],
                                       glove.embedding_vectors.shape[1],
                                       weights=[glove.embedding_vectors])(headline_input)
        headline_pooling = GlobalAveragePooling1D()(headline_embedding)

        aux_output = Dense(1, activation='sigmoid', name='aux_out')(headline_pooling)

        hour_input = Input(shape=(1,), name='hour_input')
        hour_embedding = Embedding(24, self.parameters['hour_embedding_dimensions'])(hour_input)
        hour_reshape = Reshape((self.parameters['hour_embedding_dimensions'],))(hour_embedding)

        minute_input = Input(shape=(1,), name='minute_input')
        minute_embedding = Embedding(60, self.parameters['minute_embedding_dimensions'])(minute_input)
        minute_reshape = Reshape((self.parameters['minute_embedding_dimensions'],))(minute_embedding)

        day_of_week_input = Input(shape=(1,), name='day_of_week_input')
        day_of_week_embedding = Embedding(60, self.parameters['day_of_week_embedding_dimensions'])(day_of_week_input)
        day_of_week_reshape = Reshape((self.parameters['day_of_week_embedding_dimensions'],))(day_of_week_embedding)

        day_of_year_input = Input(shape=(1,), name='day_of_year_input')
        day_of_year_embedding = Embedding(366, self.parameters['day_of_year_embedding_dimensions'])(day_of_year_input)
        day_of_year_reshape = Reshape((self.parameters['day_of_year_embedding_dimensions'],))(day_of_year_embedding)

        embedding_concatenation = concatenate([headline_pooling,
                                               hour_reshape,
                                               minute_reshape,
                                               day_of_week_reshape,
                                               day_of_year_reshape])
        relu_fully_connected = Dense(self.parameters['relu_fully_connected_dimensions'], activation='relu')(embedding_concatenation)
        batch_normalization = BatchNormalization()(relu_fully_connected)
        main_output = Dense(1, activation='sigmoid', name='main_output')(batch_normalization)

        model = Model(inputs=[headline_input,
                              hour_input,
                              minute_input,
                              day_of_week_input,
                              day_of_year_input,], outputs=[aux_output, main_output])

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy'],
                      loss_weights=[0.2, 1])

        model.summary()
        return model


class RedditModelPreprocessor:
    def __init__(self, glove, db, max_headline_length):
        self.glove = glove
        self.db = db
        self.max_headline_length = max_headline_length

        self.data = {}

    def load_data(self):
        headlines = []
        hours = []
        minutes = []
        day_of_weeks = []
        day_of_years = []
        is_top_submission = []

        labeled_articles = self.db.get_labeled_data()

        for article in labeled_articles:
            headlines.append(self.glove.text_to_sequence(article[LabelsView.HEADLINE.value]))
            hours.append(article[LabelsView.HOUR.value])
            minutes.append(article[LabelsView.MINUTE.value])
            day_of_weeks.append(article[LabelsView.DAY_OF_WEEK.value] - 1)
            day_of_years.append(article[LabelsView.DAY_OF_YEAR.value] - 1)
            is_top_submission.append(1 if article[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

        headlines = sequence.pad_sequences(headlines, maxlen=self.max_headline_length)

        self.data['headlines'] = np.array(headlines)
        self.data['hours'] = np.array(hours, dtype=int)
        self.data['minutes'] = np.array(minutes, dtype=int)
        self.data['day_of_weeks'] = np.array(day_of_weeks, dtype=int)
        self.data['day_of_years'] = np.array(day_of_years, dtype=int)
        self.data['is_top_submission'] = np.array(is_top_submission, dtype=int)
