from argparse import ArgumentParser

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, concatenate, Dense, BatchNormalization, Reshape

from src.data_handler.db_fields import LabelsView
from src.encoder.numeric_log import NumericLog
from src.models.model_builder import ModelBuilder
from src.models.preprocessor import Preprocessor
from src.utils.f1_score import precision, recall, f1
from src.utils.logging.callback_builder import CallbackBuilder
from src.utils.logging.callbacks.config_logger import ConfigLogger
from src.utils.logging.callbacks.csv_logger import CsvLogger
from src.utils.logging.callbacks.csv_plotter import CsvPlotter
from src.utils.logging.callbacks.model_saver import ModelSaver
from src.utils.settings import Settings


class TextMetricsModelBuilder(ModelBuilder):
    MODEL_IDENTIFIER = 'text_metrics_model'

    def __init__(self):
        super().__init__()

        self.required_inputs.append('headline_numeric_log')
        self.required_inputs.append('article_numeric_log')

        self.default_parameters['headline_numeric_log_embedding_dimensions'] = 5
        self.default_parameters['article_numeric_log_embedding_dimensions'] = 10

        self.default_parameters['relu_fully_connected_dimensions'] = 256
        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'
        self.default_parameters['main_output'] = 'main_output'

    def __call__(self):
        super().prepare_building()

        headline_numeric_log = self.inputs['headline_numeric_log']
        article_numeric_log = self.inputs['article_numeric_log']

        headline_word_count_input = Input(shape=(1,), name='headline_word_count_input')
        headline_word_count_embedding = Embedding(headline_numeric_log.max_log_value() + 1,
                                                  self.parameters['headline_numeric_log_embedding_dimensions'])(
            headline_word_count_input)
        headline_word_count_reshape = Reshape((self.parameters['headline_numeric_log_embedding_dimensions'],))(
            headline_word_count_embedding)

        article_word_count_input = Input(shape=(1,), name='article_word_count_input')
        article_word_count_embedding = Embedding(article_numeric_log.max_log_value() + 1,
                                                 self.parameters['article_numeric_log_embedding_dimensions'])(
            article_word_count_input)
        article_word_count_reshape = Reshape((self.parameters['article_numeric_log_embedding_dimensions'],))(
            article_word_count_embedding)

        embedding_concatenation = concatenate([headline_word_count_reshape, article_word_count_reshape])

        relu_fully_connected = Dense(self.parameters['relu_fully_connected_dimensions'], activation='relu')(
            embedding_concatenation)
        batch_normalization = BatchNormalization()(relu_fully_connected)
        main_output = Dense(1, activation='sigmoid', name=self.parameters['main_output'])(batch_normalization)

        model = Model(inputs=[headline_word_count_input, article_word_count_input], outputs=[main_output],
                      name=self.MODEL_IDENTIFIER)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1])

        model.summary()
        return model


class TextMetricsModelPreprocessor(Preprocessor):
    def __init__(self, model, headline_numeric_log, article_numeric_log):
        super().__init__(model)

        self.headline_numeric_log = headline_numeric_log
        self.article_numeric_log = article_numeric_log

    def array_to_dict(self, data):
        result = {}
        headline_numeric_logs = []
        article_numeric_logs = []
        is_top_submission = []

        output_names = [l.name for l in self.model.output_layers]

        for row in data:
            headline_numeric_logs.append(self.headline_numeric_log(row[LabelsView.HEADLINE_WORD_COUNT.value]))
            article_numeric_logs.append(self.article_numeric_log(row[LabelsView.ARTICLE_WORD_COUNT.value]))
            is_top_submission.append(1 if row[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

        result['headline_numeric_logs'] = np.array(headline_numeric_logs, dtype=int)
        result['article_numeric_logs'] = np.array(article_numeric_logs, dtype=int)
        result['is_top_submission'] = np.array(is_top_submission, dtype=int)
        result['class_weights'] = self.calculate_class_weights(result['is_top_submission'], output_names)

        return result


def train():
    settings = Settings()
    default_parameters = settings.get_training_parameter_default()

    arg_parse = ArgumentParser()
    arg_parse.add_argument('--batch_size', type=int, default=default_parameters['batch_size'])
    arg_parse.add_argument('--epochs', type=int, default=default_parameters['epochs'])
    arg_parse.add_argument('--max_headline_length', type=int, default=default_parameters['max_headline_length'])
    arg_parse.add_argument('--max_article_length', type=int, default=default_parameters['max_article_length'])

    arguments = arg_parse.parse_args()

    headline_numeric_log = NumericLog(arguments.max_headline_length)
    article_numeric_log = NumericLog(arguments.max_article_length)

    model_builder = TextMetricsModelBuilder().set_input('headline_numeric_log', headline_numeric_log).set_input(
        'article_numeric_log', article_numeric_log)
    model = model_builder()

    preprocessor = TextMetricsModelPreprocessor(model, headline_numeric_log, article_numeric_log)
    preprocessor.load_data()

    callbacks = CallbackBuilder(model, model_builder.default_parameters, arguments,
                                [CsvLogger, CsvPlotter, ConfigLogger, ModelSaver])()

    training_input = [preprocessor.training_data['headline_numeric_logs'],
                      preprocessor.training_data['article_numeric_logs']]
    training_output = [preprocessor.training_data['is_top_submission']]

    validation_input = [preprocessor.validation_data['headline_numeric_logs'],
                        preprocessor.validation_data['article_numeric_logs']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = preprocessor.training_data['class_weights']

    model.fit(training_input, training_output, batch_size=arguments.batch_size,
              epochs=arguments.epochs, callbacks=callbacks,
              validation_data=(validation_input, validation_output), class_weight=class_weights)
