from argparse import ArgumentParser

from keras import Input, Model
from keras.layers import Dense, BatchNormalization, Embedding, Reshape, Concatenate

from src.encoder.numeric_log import NumericLog
from src.models.model_builder import ModelBuilder
from src.models.preprocessor import Preprocessor
from src.utils.calculate_class_weights import calculate_class_weights
from src.utils.f1_score import precision, recall, f1
from src.utils.logging.callback_builder import CallbackBuilder
from src.utils.logging.callbacks.config_logger import ConfigLogger
from src.utils.logging.callbacks.csv_logger import CsvLogger
from src.utils.logging.callbacks.csv_plotter import CsvPlotter
from src.utils.logging.callbacks.model_saver import ModelSaver
from src.utils.settings import Settings


class Model7Builder(ModelBuilder):
    def __init__(self):
        super().__init__()

        self.default_parameters['fully_connected_dimensions'] = 32
        self.default_parameters['fully_connected_activation'] = 'tanh'

        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'

    def __call__(self):
        super().prepare_building()

        competitive_score_input = Input(shape=(1,), name='competitive_score_input')
        competitive_score_sigmoid = Dense(1, activation='sigmoid')(competitive_score_input)
        fully_connected = Dense(self.parameters['fully_connected_dimensions'],
                                activation=self.parameters['fully_connected_activation'])(competitive_score_sigmoid)
        batch_normalization = BatchNormalization()(fully_connected)
        main_output = Dense(1, activation='sigmoid', name='output')(batch_normalization)

        model = Model(inputs=[competitive_score_input],
                      outputs=[main_output],
                      name=self.model_identifier)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1])

        return model

    @property
    def model_identifier(self):
        return 'model_7'


def train():
    settings = Settings()
    default_parameters = settings.get_training_parameters()

    arg_parse = ArgumentParser()
    arg_parse.add_argument('--batch_size', type=int, default=default_parameters['batch_size'])
    arg_parse.add_argument('--epochs', type=int, default=default_parameters['epochs'])

    arg_parse.add_argument('--fully_connected_dimensions', type=int)
    arg_parse.add_argument('--fully_connected_activation', type=str)

    arg_parse.add_argument('--optimizer', type=str)
    arg_parse.add_argument('--loss', type=str)
    arguments = arg_parse.parse_args()

    model_builder = Model7Builder()

    for key in model_builder.default_parameters.keys():
        if hasattr(arguments, key) and getattr(arguments, key):
            model_builder.set_parameter(key, getattr(arguments, key))

    model = model_builder()

    preprocessor = Preprocessor(model)

    preprocessor.load_data(['competitive_score', 'is_top_submission'])
    training_input = [preprocessor.training_data['competitive_score']]
    validation_input = [preprocessor.validation_data['competitive_score']]
    training_output = [preprocessor.training_data['is_top_submission']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = calculate_class_weights(preprocessor.training_data['is_top_submission'],
                                            [ol.name for ol in model.output_layers])

    callbacks = CallbackBuilder(model, model_builder.default_parameters, arguments,
                                [CsvLogger, CsvPlotter, ConfigLogger, ModelSaver])()

    model.fit(training_input, training_output, batch_size=arguments.batch_size, epochs=arguments.epochs,
              callbacks=callbacks, validation_data=(validation_input, validation_output), class_weight=class_weights)
