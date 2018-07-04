from argparse import ArgumentParser

import numpy as np
from keras import Input, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, BatchNormalization
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


class HeadlineModelBuilder(ModelBuilder):
    MODEL_IDENTIFIER = 'headline_model'

    def __init__(self):
        super().__init__()

        self.required_inputs.append('glove')
        self.required_parameters.append('max_headline_length')

        self.default_parameters['relu_fully_connected_dimensions'] = 256
        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'
        self.default_parameters['main_output'] = 'main_output'

    def __call__(self):
        super().prepare_building()

        glove = self.inputs['glove']
        headline_input = Input(shape=(self.parameters['max_headline_length'],), name='headline_input')
        headline_embedding = Embedding(glove.embedding_vectors.shape[0],
                                       glove.embedding_vectors.shape[1],
                                       weights=[glove.embedding_vectors])(headline_input)
        headline_pooling = GlobalAveragePooling1D()(headline_embedding)

        relu_fully_connected = Dense(self.parameters['relu_fully_connected_dimensions'], activation='relu')(
            headline_pooling)
        batch_normalization = BatchNormalization()(relu_fully_connected)
        main_output = Dense(1, activation='sigmoid', name=self.parameters['main_output'])(batch_normalization)

        model = Model(inputs=[headline_input], outputs=[main_output], name=self.MODEL_IDENTIFIER)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1])

        model.summary()
        return model


class HeadlinePreprocessor(Preprocessor):

    def __init__(self, model, glove, max_headline_length):
        super().__init__(model)
        self.glove = glove
        self.max_headline_length = max_headline_length

    def array_to_dict(self, data):
        result = {}
        headlines = []
        is_top_submission = []

        output_names = [l.name for l in self.model.output_layers]

        for article in data:
            headlines.append(self.glove.text_to_sequence(article[LabelsView.HEADLINE.value]))
            is_top_submission.append(1 if article[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

        headlines = sequence.pad_sequences(headlines, maxlen=self.max_headline_length)

        result['headlines'] = np.array(headlines)
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
    arg_parse.add_argument('--max_headline_length', type=int, default=default_parameters['max_headline_length'])

    arguments = arg_parse.parse_args()

    glove = Glove(arguments.dictionary_size)
    glove.load_embedding()

    model_builder = HeadlineModelBuilder() \
        .set_input('glove', glove) \
        .set_parameter('max_headline_length', arguments.max_headline_length)

    model = model_builder()

    preprocessor = HeadlinePreprocessor(model, glove, arguments.max_headline_length)
    preprocessor.load_data()

    callbacks = CallbackBuilder(model, arguments, [CsvLogger, CsvPlotter, ConfigLogger])()

    training_input = [preprocessor.training_data['headlines']]
    training_output = [preprocessor.training_data['is_top_submission']]

    validation_input = [preprocessor.validation_data['headlines']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = preprocessor.training_data['class_weights']

    model.fit(training_input, training_output, batch_size=arguments.batch_size, epochs=arguments.epochs,
              callbacks=callbacks,
              validation_data=(validation_input, validation_output), class_weight=class_weights)
