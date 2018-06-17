import numpy as np
from keras import Input, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, BatchNormalization, Multiply, RepeatVector, \
    Permute
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from src.data_handler.db_fields import LabelsView
from src.data_handler.labels_db import LabelsDb
from src.encoder.glove import Glove
from src.encoder.tf_idf import TfIdf
from src.models.model_builder import ModelBuilder
from src.models.preprocessor import Preprocessor
from src.utils.f1_score import f1, precision, recall
from src.utils.logging.callback_builder import CallbackBuilder
from src.utils.logging.callbacks.config_logger import ConfigLogger
from src.utils.logging.callbacks.csv_logger import CsvLogger
from src.utils.logging.callbacks.csv_plotter import CsvPlotter


class HeadlineTfIdfModelBuilder(ModelBuilder):
    MODEL_IDENTIFIER = 'headline_tf_idf_model'

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
        headline_embedding = Embedding(glove.weights_matrix.shape[0],
                                       glove.weights_matrix.shape[1],
                                       weights=[glove.weights_matrix])(headline_input)

        tf_idf_input = Input(shape=(self.parameters['max_headline_length'],), name='tf_idf_input')

        tf_idf_reshape = RepeatVector(glove.weights_matrix.shape[1])(tf_idf_input)
        tf_idf_reshape = Permute((2, 1))(tf_idf_reshape)

        tf_idf_embedding = Multiply()([headline_embedding, tf_idf_reshape])

        headline_pooling = GlobalAveragePooling1D()(tf_idf_embedding)

        relu_fully_connected = Dense(self.parameters['relu_fully_connected_dimensions'], activation='relu')(
            headline_pooling)
        batch_normalization = BatchNormalization()(relu_fully_connected)
        main_output = Dense(1, activation='sigmoid', name=self.parameters['main_output'])(batch_normalization)

        model = Model(inputs=[headline_input, tf_idf_input], outputs=[main_output], name=self.MODEL_IDENTIFIER)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1])

        model.summary()
        return model


class HeadlineTfIdfPreprocessor(Preprocessor):

    def __init__(self, model, tokenizer, tf_idf, max_headline_length):
        super().__init__(model)
        self.tokenizer = tokenizer
        self.tf_idf = tf_idf
        self.max_headline_length = max_headline_length

    def array_to_dict(self, data):
        result = {}
        headlines = []
        is_top_submission = []

        output_names = [l.name for l in self.model.output_layers]

        for article in data:
            headlines.append(article[LabelsView.HEADLINE.value])
            is_top_submission.append(1 if article[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

        headlines = self.tokenizer.texts_to_sequences(headlines)
        headlines = sequence.pad_sequences(headlines, maxlen=self.max_headline_length)
        tf_idf = self.tf_idf.get_tf_idf(headlines)

        result['headlines'] = np.array(headlines)
        result['tf_idf'] = np.array(tf_idf)
        result['is_top_submission'] = np.array(is_top_submission, dtype=int)
        result['class_weights'] = self.calculate_class_weights(result['is_top_submission'], output_names)

        return result


def train():
    hyper_parameters = {}

    hyper_parameters['dictionary_size'] = 40000
    hyper_parameters['max_headline_length'] = 20
    hyper_parameters['batch_size'] = 64
    hyper_parameters['epochs'] = 20

    headline_corpus = [article[LabelsView.HEADLINE.value] for article in LabelsDb().get_labeled_data()]

    tokenizer = Tokenizer(hyper_parameters['dictionary_size'])
    tokenizer.fit_on_texts(headline_corpus)

    glove = Glove(tokenizer)
    glove.load_embedding()

    tf_idf = TfIdf(tokenizer)

    model_builder = HeadlineTfIdfModelBuilder() \
        .set_input('glove', glove) \
        .set_parameter('max_headline_length', hyper_parameters['max_headline_length'])

    model = model_builder()

    preprocessor = HeadlineTfIdfPreprocessor(model, tokenizer, tf_idf, hyper_parameters['max_headline_length'])
    preprocessor.load_data()

    callbacks = CallbackBuilder(model, hyper_parameters, [CsvLogger, CsvPlotter, ConfigLogger])()

    training_input = [preprocessor.training_data['headlines'], preprocessor.training_data['tf_idf']]
    training_output = [preprocessor.training_data['is_top_submission']]

    validation_input = [preprocessor.validation_data['headlines'], preprocessor.validation_data['tf_idf']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = preprocessor.training_data['class_weights']

    model.fit(training_input, training_output, batch_size=hyper_parameters['batch_size'],
              epochs=hyper_parameters['epochs'], callbacks=callbacks,
              validation_data=(validation_input, validation_output), class_weight=class_weights)
