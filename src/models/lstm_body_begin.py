from argparse import ArgumentParser

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPooling1D, \
    LSTM
from keras.preprocessing import sequence

from src.data_handler.db_fields import LabelsView
from src.encoder.glove import Glove
from src.models.model_builder import ModelBuilder
from src.models.preprocessor import Preprocessor
from src.utils.f1_score import f1, precision, recall
from src.utils.logging.callback_builder import CallbackBuilder
from src.utils.logging.callbacks.config_logger import ConfigLogger
from src.utils.logging.callbacks.csv_logger import CsvLogger
from src.utils.logging.callbacks.csv_plotter import CsvPlotter
from src.utils.settings import Settings


class LstmBodyBeginModelBuilder(ModelBuilder):
    MODEL_IDENTIFIER = 'lstm_body_begin'

    def __init__(self):
        super().__init__()

        self.required_inputs.append('glove')
        self.required_parameters.append('body_begin_length')

        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'
        self.default_parameters['main_output'] = 'main_output'
        self.default_parameters['lstm_units'] = 100

    def __call__(self):
        super().prepare_building()

        glove = self.inputs['glove']

        body_begin_input = Input(shape=(self.parameters['body_begin_length'],), name='body_begin_input')

        headline_embedding = Embedding(glove.embedding_vectors.shape[0],
                                       glove.embedding_vectors.shape[1],
                                       weights=[glove.embedding_vectors],
                                       trainable=False)(body_begin_input)

        lstm = LSTM(self.default_parameters['lstm_units'])(headline_embedding)

        main_out = Dense(1, activation='sigmoid', name=self.default_parameters['main_output'])(lstm)

        model = Model(inputs=[body_begin_input], outputs=main_out, name=self.MODEL_IDENTIFIER)

        model.compile(loss=self.default_parameters['loss'], optimizer=self.default_parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1])

        model.summary()
        return model


class LstmBodyBeginPreprocessor(Preprocessor):
    def __init__(self, model, glove, body_begin_length):
        super().__init__(model)
        self.glove = glove
        self.body_begin_length = body_begin_length

    def array_to_dict(self, data):
        result = {}
        body_beginnings = []
        is_top_submission = []

        output_names = [l.name for l in self.model.output_layers]

        for article in data:
            body_begin = self.glove.text_to_sequence(article[LabelsView.ARTICLE.value], limit=self.body_begin_length)
            body_beginnings.append(body_begin)
            is_top_submission.append(1 if article[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

        body_beginnings = sequence.pad_sequences(body_beginnings, maxlen=self.body_begin_length)

        result['body_beginnings'] = np.array(body_beginnings)
        result['is_top_submission'] = np.array(is_top_submission, dtype=int)
        result['class_weights'] = self.calculate_class_weights(result['is_top_submission'], output_names)

        return result


def train():
    settings = Settings()
    default_parameters = settings.get_training_parameter_default()

    arg_parse = ArgumentParser()
    arg_parse.add_argument('--batch_size', type=int, default=default_parameters['batch_size'])
    arg_parse.add_argument('--epochs', type=int, default=default_parameters['epochs'])
    arg_parse.add_argument('--dictionary_size', type=int, default=default_parameters['dictionary_size'])
    arg_parse.add_argument('--body_begin_length', type=int, default=default_parameters['body_begin_length'])

    arguments = arg_parse.parse_args()

    glove = Glove(arguments.dictionary_size)
    glove.load_embedding()

    model_builder = LstmBodyBeginModelBuilder() \
        .set_input('glove', glove) \
        .set_parameter('body_begin_length', arguments.body_begin_length)

    model = model_builder()

    preprocessor = LstmBodyBeginPreprocessor(model, glove, arguments.body_begin_length)
    preprocessor.load_data()

    callbacks = CallbackBuilder(model, arguments, [CsvLogger, CsvPlotter, ConfigLogger])()

    training_input = [preprocessor.training_data['body_beginnings']]
    training_output = [preprocessor.training_data['is_top_submission']]

    validation_input = [preprocessor.validation_data['body_beginnings']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = preprocessor.training_data['class_weights']

    model.fit(training_input, training_output, batch_size=arguments.batch_size, epochs=arguments.epochs,
              callbacks=callbacks,
              validation_data=(validation_input, validation_output), class_weight=class_weights)
