from argparse import ArgumentParser

from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, \
    Concatenate, LSTM

from src.encoder.glove import Glove
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


class Model23Builder(ModelBuilder):
    def __init__(self):
        super().__init__()

        self.required_inputs.append('glove')
        self.required_parameters.append('max_headline_length')
        self.required_parameters.append('body_begin_length')

        self.default_parameters['filter_count_5'] = 5
        self.default_parameters['filter_count_3'] = 5
        self.default_parameters['filter_count_1'] = 5

        self.default_parameters['lstm_units'] = 64

        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'

    def __call__(self):
        super().prepare_building()

        glove = self.inputs['glove']
        headline_input = Input(shape=(self.parameters['max_headline_length'],), name='headline_input')
        headline_embedding = Embedding(glove.embedding_vectors.shape[0],
                                       glove.embedding_vectors.shape[1],
                                       weights=[glove.embedding_vectors])(headline_input)

        convolution_5 = Conv1D(self.parameters['filter_count_5'], kernel_size=5)(headline_embedding)
        convolution_5_max = GlobalMaxPooling1D()(convolution_5)
        convolution_3 = Conv1D(self.parameters['filter_count_3'], kernel_size=3)(headline_embedding)
        convolution_3_max = GlobalMaxPooling1D()(convolution_3)
        convolution_1 = Conv1D(self.parameters['filter_count_1'], kernel_size=1)(headline_embedding)
        convolution_1_max = GlobalMaxPooling1D()(convolution_1)

        body_begin_input = Input(shape=(self.parameters['body_begin_length'],), name='body_begin_input')
        body_begin_embedding = Embedding(glove.embedding_vectors.shape[0],
                                         glove.embedding_vectors.shape[1],
                                         weights=[glove.embedding_vectors],
                                         trainable=False)(body_begin_input)
        lstm = LSTM(self.parameters['lstm_units'])(body_begin_embedding)

        concat = Concatenate()([convolution_5_max, convolution_3_max, convolution_1_max, lstm])
        main_output = Dense(1, activation='sigmoid', name='output')(concat)

        model = Model(inputs=[headline_input, body_begin_input], outputs=[main_output], name=self.model_identifier)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1])

        return model

    @property
    def model_identifier(self):
        return 'model_23'


def train():
    settings = Settings()
    default_parameters = settings.get_training_parameter_default()

    arg_parse = ArgumentParser()
    arg_parse.add_argument('--batch_size', type=int, default=default_parameters['batch_size'])
    arg_parse.add_argument('--epochs', type=int, default=default_parameters['epochs'])

    arg_parse.add_argument('--dictionary_size', type=int, default=default_parameters['dictionary_size'])
    arg_parse.add_argument('--max_headline_length', type=int, default=default_parameters['max_headline_length'])
    arg_parse.add_argument('--body_begin_length', type=int, default=default_parameters['body_begin_length'])

    arg_parse.add_argument('--filter_count_5', type=int)
    arg_parse.add_argument('--filter_count_3', type=int)
    arg_parse.add_argument('--filter_count_1', type=int)

    arg_parse.add_argument('--lstm_units', type=int)

    arg_parse.add_argument('--optimizer', type=str)
    arg_parse.add_argument('--loss', type=str)
    arguments = arg_parse.parse_args()

    glove = Glove(arguments.dictionary_size)
    glove.load_embedding()

    model_builder = Model23Builder() \
        .set_input('glove', glove) \
        .set_parameter('max_headline_length', arguments.max_headline_length)\
        .set_parameter('body_begin_length', arguments.body_begin_length)

    for key in model_builder.default_parameters.keys():
        if getattr(arguments, key):
            model_builder.set_parameter(key, getattr(arguments, key))

    model = model_builder()

    preprocessor = Preprocessor(model)
    preprocessor.set_encoder('glove', glove)
    preprocessor.set_parameter('max_headline_length', arguments.max_headline_length)
    preprocessor.set_parameter('body_begin_length', arguments.body_begin_length)

    preprocessor.load_data(['headline', 'body_begin', 'is_top_submission'])

    training_input = [preprocessor.training_data[key] for key in ['headline', 'body_begin']]
    validation_input = [preprocessor.validation_data[key] for key in ['headline', 'body_begin']]
    training_output = [preprocessor.training_data['is_top_submission']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = calculate_class_weights(preprocessor.training_data['is_top_submission'],
                                            [ol.name for ol in model.output_layers])

    callbacks = CallbackBuilder(model, model_builder.default_parameters, arguments,
                                [CsvLogger, CsvPlotter, ConfigLogger, ModelSaver])()

    model.fit(training_input, training_output, batch_size=arguments.batch_size, epochs=arguments.epochs,
              callbacks=callbacks, validation_data=(validation_input, validation_output), class_weight=class_weights)
