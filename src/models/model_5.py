from argparse import ArgumentParser

from keras import Input, Model
from keras.layers import Dense, BatchNormalization

from src.encoder.doc2vec import Doc2Vec
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


class Model5Builder(ModelBuilder):
    def __init__(self):
        super().__init__()

        self.required_inputs.append('article_doc2vec')

        self.default_parameters['fully_connected_dimensions'] = 128
        self.default_parameters['fully_connected_activation'] = 'tanh'

        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'

    def __call__(self):
        super().prepare_building()

        doc2vec = self.inputs['article_doc2vec']

        article_input = Input(shape=(doc2vec.get_dimensions(),), name='article_input')
        fully_connected = Dense(self.parameters['fully_connected_dimensions'],
                                activation=self.parameters['fully_connected_activation'])(article_input)
        batch_normalization = BatchNormalization()(fully_connected)
        main_output = Dense(1, activation='sigmoid', name='output')(batch_normalization)

        model = Model(inputs=[article_input], outputs=[main_output], name=self.model_identifier)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1])

        return model

    @property
    def model_identifier(self):
        return 'model_5'


def train():
    settings = Settings()
    default_parameters = settings.get_training_parameter_default()
    choices_parameters = settings.get_training_parameter_choices()

    arg_parse = ArgumentParser()
    arg_parse.add_argument('--batch_size', type=int, default=default_parameters['batch_size'])
    arg_parse.add_argument('--epochs', type=int, default=default_parameters['epochs'])

    arg_parse.add_argument('--article_doc2vec_dimensions', type=int,
                           default=default_parameters['article_doc2vec_dimensions'],
                           choices=choices_parameters['article_doc2vec_dimensions'])

    arg_parse.add_argument('--fully_connected_dimensions', type=int)
    arg_parse.add_argument('--fully_connected_activation', type=str)

    arg_parse.add_argument('--optimizer', type=str)
    arg_parse.add_argument('--loss', type=str)
    arguments = arg_parse.parse_args()

    article_doc2vec = Doc2Vec()
    article_doc2vec.load_model('article', arguments.article_doc2vec_dimensions)

    model_builder = Model5Builder().set_input('article_doc2vec', article_doc2vec)

    for key in model_builder.default_parameters.keys():
        if getattr(arguments, key):
            model_builder.set_parameter(key, getattr(arguments, key))

    model = model_builder()

    preprocessor = Preprocessor(model)
    preprocessor.set_encoder('article_doc2vec', article_doc2vec)

    preprocessor.load_data(['article_vector', 'is_top_submission'])

    training_input = [preprocessor.training_data['article_vector']]
    validation_input = [preprocessor.validation_data['article_vector']]
    training_output = [preprocessor.training_data['is_top_submission']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = calculate_class_weights(preprocessor.training_data['is_top_submission'],
                                            [ol.name for ol in model.output_layers])

    callbacks = CallbackBuilder(model, model_builder.default_parameters, arguments,
                                [CsvLogger, CsvPlotter, ConfigLogger, ModelSaver])()

    model.fit(training_input, training_output, batch_size=arguments.batch_size, epochs=arguments.epochs,
              callbacks=callbacks, validation_data=(validation_input, validation_output), class_weight=class_weights)
