import numpy as np
from keras import Input, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Reshape, concatenate, BatchNormalization
from keras.preprocessing import sequence
from src.utils.logging.callback_builder import CallbackBuilder

from src.data_handler.db_fields import LabelsView
from src.encoder.glove import Glove
from src.models.model_builder import ModelBuilder
from src.models.preprocessor import Preprocessor
from src.utils.f1_score import f1, precision, recall
from src.utils.logging.callbacks.config_logger import ConfigLogger
from src.utils.logging.callbacks.csv_logger import CsvLogger
from src.utils.logging.callbacks.csv_plotter import CsvPlotter


class HeadlineTimeModelBuilder(ModelBuilder):

    MODEL_IDENTIFIER = 'headline_time_model'

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
        self.default_parameters['headline_output'] = 'headline_output'
        self.default_parameters['main_output'] = 'main_output'

    def __call__(self):
        super().prepare_building()

        glove = self.inputs['glove']
        headline_input = Input(shape=(self.parameters['max_headline_length'],), name='headline_input')
        headline_embedding = Embedding(glove.embedding_vectors.shape[0],
                                       glove.embedding_vectors.shape[1],
                                       weights=[glove.embedding_vectors])(headline_input)
        headline_pooling = GlobalAveragePooling1D()(headline_embedding)

        headline_output = Dense(1, activation='sigmoid', name=self.parameters['headline_output'])(headline_pooling)

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
        relu_fully_connected = Dense(self.parameters['relu_fully_connected_dimensions'], activation='relu')(
            embedding_concatenation)
        batch_normalization = BatchNormalization()(relu_fully_connected)
        main_output = Dense(1, activation='sigmoid', name=self.parameters['main_output'])(batch_normalization)

        model = Model(inputs=[headline_input,
                              hour_input,
                              minute_input,
                              day_of_week_input,
                              day_of_year_input, ], outputs=[headline_output, main_output], name=self.MODEL_IDENTIFIER)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1],
                      loss_weights=[0.2, 1])

        model.summary()
        return model


class HeadlineTimePreprocessor(Preprocessor):

    def __init__(self, model, glove, max_headline_length):
        super().__init__(model)
        self.glove = glove
        self.max_headline_length = max_headline_length

    def array_to_dict(self, data):
        result = {}
        headlines = []
        hours = []
        minutes = []
        day_of_weeks = []
        day_of_years = []
        is_top_submission = []

        output_names = [l.name for l in self.model.output_layers]

        for article in data:
            headlines.append(self.glove.text_to_sequence(article[LabelsView.HEADLINE.value]))
            hours.append(article[LabelsView.HOUR.value])
            minutes.append(article[LabelsView.MINUTE.value])
            day_of_weeks.append(article[LabelsView.DAY_OF_WEEK.value])
            day_of_years.append(article[LabelsView.DAY_OF_YEAR.value] - 1)
            is_top_submission.append(1 if article[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

        headlines = sequence.pad_sequences(headlines, maxlen=self.max_headline_length)

        result['headlines'] = np.array(headlines)
        result['hours'] = np.array(hours, dtype=int)
        result['minutes'] = np.array(minutes, dtype=int)
        result['day_of_weeks'] = np.array(day_of_weeks, dtype=int)
        result['day_of_years'] = np.array(day_of_years, dtype=int)
        result['is_top_submission'] = np.array(is_top_submission, dtype=int)
        result['class_weights'] = self.calculate_class_weights(result['is_top_submission'], output_names)

        return result


def train():
    hyper_parameters = {}

    hyper_parameters['dictionary_size'] = 40000
    hyper_parameters['max_headline_length'] = 20
    hyper_parameters['batch_size'] = 64
    hyper_parameters['epochs'] = 20

    glove = Glove(hyper_parameters['dictionary_size'])
    glove.load_embedding()

    model_builder = HeadlineTimeModelBuilder() \
        .set_input('glove', glove) \
        .set_parameter('max_headline_length', hyper_parameters['max_headline_length'])

    model = model_builder()

    preprocessor = HeadlineTimePreprocessor(model, glove, hyper_parameters['max_headline_length'])
    preprocessor.load_data()

    callbacks = CallbackBuilder(model, hyper_parameters, [CsvLogger, CsvPlotter, ConfigLogger])()

    training_input = [preprocessor.training_data['headlines'],
                      preprocessor.training_data['hours'],
                      preprocessor.training_data['minutes'],
                      preprocessor.training_data['day_of_weeks'],
                      preprocessor.training_data['day_of_years']]

    training_output = [preprocessor.training_data['is_top_submission'], preprocessor.training_data['is_top_submission']]

    validation_input = [preprocessor.validation_data['headlines'],
                        preprocessor.validation_data['hours'],
                        preprocessor.validation_data['minutes'],
                        preprocessor.validation_data['day_of_weeks'],
                        preprocessor.validation_data['day_of_years']]

    validation_output = [preprocessor.validation_data['is_top_submission'],
                         preprocessor.validation_data['is_top_submission']]

    class_weights = preprocessor.training_data['class_weights']

    model.fit(training_input, training_output, batch_size=hyper_parameters['batch_size'], epochs=hyper_parameters['epochs'], callbacks=callbacks,
              validation_data=(validation_input, validation_output), class_weight=class_weights)
