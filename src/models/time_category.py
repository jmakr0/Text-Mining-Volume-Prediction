from argparse import ArgumentParser

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Dense, Reshape, concatenate, BatchNormalization

from src.data_handler.db_fields import LabelsView
from src.models.model_builder import ModelBuilder
from src.models.preprocessor import Preprocessor
from src.utils.f1_score import f1, precision, recall
from src.utils.logging.callback_builder import CallbackBuilder
from src.utils.logging.callbacks.config_logger import ConfigLogger
from src.utils.logging.callbacks.csv_logger import CsvLogger
from src.utils.logging.callbacks.csv_plotter import CsvPlotter
from src.utils.settings import Settings


class TimeCategoryModelBuilder(ModelBuilder):
    MODEL_IDENTIFIER = 'time_category_model'

    def __init__(self):
        super().__init__()

        self.default_parameters['hour_embedding_dimensions'] = 5
        self.default_parameters['minute_embedding_dimensions'] = 5
        self.default_parameters['day_of_week_embedding_dimensions'] = 5
        self.default_parameters['day_of_year_embedding_dimensions'] = 5
        self.default_parameters['category_embedding_dimensions'] = 5
        self.default_parameters['relu_fully_connected_dimensions'] = 128
        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'
        self.default_parameters['main_output'] = 'main_output'

    def __call__(self):
        super().prepare_building()

        hour_input = Input(shape=(1,), name='hour_input')
        hour_embedding = Embedding(24, self.parameters['hour_embedding_dimensions'])(hour_input)
        hour_reshape = Reshape((self.parameters['hour_embedding_dimensions'],))(hour_embedding)

        minute_input = Input(shape=(1,), name='minute_input')
        minute_embedding = Embedding(60, self.parameters['minute_embedding_dimensions'])(minute_input)
        minute_reshape = Reshape((self.parameters['minute_embedding_dimensions'],))(minute_embedding)

        day_of_week_input = Input(shape=(1,), name='day_of_week_input')
        day_of_week_embedding = Embedding(7, self.parameters['day_of_week_embedding_dimensions'])(day_of_week_input)
        day_of_week_reshape = Reshape((self.parameters['day_of_week_embedding_dimensions'],))(day_of_week_embedding)

        day_of_year_input = Input(shape=(1,), name='day_of_year_input')
        day_of_year_embedding = Embedding(366, self.parameters['day_of_year_embedding_dimensions'])(day_of_year_input)
        day_of_year_reshape = Reshape((self.parameters['day_of_year_embedding_dimensions'],))(day_of_year_embedding)

        category_input = Input(shape=(1,), name='category_input')
        # the category is saved as an ID in [1,80]
        category_embedding = Embedding(81, self.parameters['category_embedding_dimensions'])(category_input)
        category_reshape = Reshape((self.parameters['category_embedding_dimensions'],))(category_embedding)

        embedding_concatenation = concatenate([hour_reshape,
                                               minute_reshape,
                                               day_of_week_reshape,
                                               day_of_year_reshape,
                                               category_reshape])

        relu_fully_connected = Dense(self.parameters['relu_fully_connected_dimensions'], activation='relu')(
            embedding_concatenation)
        batch_normalization = BatchNormalization()(relu_fully_connected)
        main_output = Dense(1, activation='sigmoid', name=self.parameters['main_output'])(batch_normalization)

        model = Model(inputs=[hour_input,
                              minute_input,
                              day_of_week_input,
                              day_of_year_input,
                              category_input], outputs=[main_output], name=self.MODEL_IDENTIFIER)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1],
                      )

        model.summary()
        return model


class TimeCategoryPreprocessor(Preprocessor):

    def __init__(self, model):
        super().__init__(model)

    def array_to_dict(self, data):
        result = {}
        hours = []
        minutes = []
        day_of_weeks = []
        day_of_years = []
        is_top_submission = []
        category_ids = []

        output_names = [l.name for l in self.model.output_layers]

        for article in data:
            hours.append(article[LabelsView.HOUR.value])
            minutes.append(article[LabelsView.MINUTE.value])
            day_of_weeks.append(article[LabelsView.DAY_OF_WEEK.value])
            day_of_years.append(article[LabelsView.DAY_OF_YEAR.value] - 1)
            category_ids.append(article[LabelsView.CATEGORY_ID.value])
            is_top_submission.append(1 if article[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

        result['hours'] = np.array(hours, dtype=int)
        result['minutes'] = np.array(minutes, dtype=int)
        result['day_of_weeks'] = np.array(day_of_weeks, dtype=int)
        result['day_of_years'] = np.array(day_of_years, dtype=int)
        result['is_top_submission'] = np.array(is_top_submission, dtype=int)
        result['category_ids'] = np.array(category_ids, dtype=int)
        result['class_weights'] = self.calculate_class_weights(result['is_top_submission'], output_names)

        return result


def train():
    settings = Settings()
    default_parameters = settings.get_training_parameter_default()

    arg_parse = ArgumentParser()
    arg_parse.add_argument('--batch_size', type=int, default=default_parameters['batch_size'])
    arg_parse.add_argument('--epochs', type=int, default=default_parameters['epochs'])
    arg_parse.add_argument('--dictionary_size', type=int, default=default_parameters['dictionary_size'])

    arguments = arg_parse.parse_args()

    model_builder = TimeCategoryModelBuilder()
    model = model_builder()

    preprocessor = TimeCategoryPreprocessor(model)
    preprocessor.load_data()

    callbacks = CallbackBuilder(model, model_builder.default_parameters, arguments, [CsvLogger, CsvPlotter, ConfigLogger])()

    training_input = [preprocessor.training_data['hours'],
                      preprocessor.training_data['minutes'],
                      preprocessor.training_data['day_of_weeks'],
                      preprocessor.training_data['day_of_years'],
                      preprocessor.training_data['category_ids']]

    training_output = preprocessor.training_data['is_top_submission']

    validation_input = [preprocessor.validation_data['hours'],
                        preprocessor.validation_data['minutes'],
                        preprocessor.validation_data['day_of_weeks'],
                        preprocessor.validation_data['day_of_years'],
                        preprocessor.validation_data['category_ids']]

    validation_output = preprocessor.validation_data['is_top_submission']

    class_weights = preprocessor.training_data['class_weights']

    model.fit(training_input,
              training_output,
              batch_size=arguments.batch_size,
              epochs=arguments.epochs,
              callbacks=callbacks,
              validation_data=(validation_input, validation_output),
              class_weight=class_weights)
