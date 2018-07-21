from argparse import ArgumentParser

from keras import Input, Model
from keras.layers import Dense, BatchNormalization, Embedding, Reshape

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


class Model4Builder(ModelBuilder):
    def __init__(self):
        super().__init__()
        settings = Settings()

        self.default_parameters['category_embedding_dimensions'] = settings.get_network_parameters('category_embedding_dimensions')
        self.default_parameters['fully_connected_dimensions'] = settings.get_network_parameters('fully_connected_dimensions')
        self.default_parameters['fully_connected_activation'] = settings.get_network_parameters('fully_connected_activation')

        self.default_parameters['optimizer'] = settings.get_network_parameters('optimizer')
        self.default_parameters['loss'] = settings.get_network_parameters('loss_function')

    def __call__(self):
        super().prepare_building()

        category_input = Input(shape=(1,), name='category_input')
        category_embedding = Embedding(81, self.parameters['category_embedding_dimensions'])(category_input)
        category_reshape = Reshape((self.parameters['category_embedding_dimensions'],))(category_embedding)

        fully_connected = Dense(self.parameters['fully_connected_dimensions'],
                                activation=self.parameters['fully_connected_activation'])(category_reshape)
        batch_normalization = BatchNormalization()(fully_connected)
        main_output = Dense(1, activation='sigmoid', name='output')(batch_normalization)

        model = Model(inputs=[category_input], outputs=[main_output], name=self.model_identifier)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1])

        return model

    @property
    def model_identifier(self):
        return 'model_4'


def train():
    settings = Settings()

    batch_size = settings.get_training_parameters('batch_size')
    epochs = settings.get_training_parameters('epochs')

    model_builder = Model4Builder()

    model = model_builder()

    preprocessor = Preprocessor(model)

    preprocessor.load_data(['category', 'is_top_submission'])

    training_input = [preprocessor.training_data['category']]
    validation_input = [preprocessor.validation_data['category']]
    training_output = [preprocessor.training_data['is_top_submission']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = calculate_class_weights(preprocessor.training_data['is_top_submission'],
                                            [ol.name for ol in model.output_layers])

    callbacks = CallbackBuilder(model, [CsvLogger, CsvPlotter, ConfigLogger, ModelSaver])()

    model.fit(training_input, training_output, batch_size=batch_size, epochs=epochs,
              callbacks=callbacks, validation_data=(validation_input, validation_output), class_weight=class_weights)
