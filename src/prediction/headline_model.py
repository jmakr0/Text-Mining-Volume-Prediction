import numpy as np
from keras import Input, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, BatchNormalization
from keras.preprocessing import sequence

from src.data_handler.db_fields import LabelsView
from src.models.glove import Glove
from src.prediction.model_builder import ModelBuilder
from src.prediction.preprocessor import Preprocessor
from src.utils.config_logger import ConfigLoggerCallback
from src.utils.csv_plot import CSVPlotterCallback
from src.utils.custom_csv_logger import CustomCsvLogger
from src.utils.f1_score import f1, precision, recall
from src.utils.settings import Settings
from src.utils.utils import get_timestamp


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

        relu_fully_connected = Dense(self.parameters['relu_fully_connected_dimensions'], activation='relu')(headline_pooling)
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
    hyper_parameters = {}

    hyper_parameters['dictionary_size'] = 40000
    hyper_parameters['max_headline_length'] = 20
    hyper_parameters['batch_size'] = 64
    hyper_parameters['epochs'] = 2

    glove = Glove(hyper_parameters['dictionary_size'])
    glove.load_embedding()

    model_builder = HeadlineModelBuilder() \
        .set_input('glove', glove) \
        .set_parameter('max_headline_length', hyper_parameters['max_headline_length'])

    model = model_builder()

    preprocessor = HeadlinePreprocessor(model, glove, hyper_parameters['max_headline_length'])
    preprocessor.load_data()

    timestamp = get_timestamp()

    csv_logger = CustomCsvLogger(model.name, timestamp)

    plot_config = [('f1', (0.1, 0.0, 0.9), 'f1-score'), ('val_f1', 'g', 'validation f1-score')]
    plot_callback = CSVPlotterCallback(csv_logger.csv_file, plot_config)

    config_log_callback = ConfigLoggerCallback(model, timestamp, model_builder.get_model_description(), hyper_parameters)

    training_input = [preprocessor.training_data['headlines']]
    training_output = [preprocessor.training_data['is_top_submission']]

    validation_input = [preprocessor.validation_data['headlines']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = preprocessor.training_data['class_weights']

    model.fit(training_input, training_output, batch_size=hyper_parameters['batch_size'], epochs=hyper_parameters['epochs'], callbacks=[csv_logger, plot_callback, config_log_callback],
              validation_data=(validation_input, validation_output), class_weight=class_weights)
