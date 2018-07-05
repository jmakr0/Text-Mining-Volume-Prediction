import numpy as np
from keras import Input, Model
from keras.layers import Dense, BatchNormalization

from src.data_handler.db_fields import LabelsView
from src.encoder.doc2vec import Doc2Vec
from src.models.model_builder import ModelBuilder
from src.models.preprocessor import Preprocessor
from src.utils.f1_score import f1, precision, recall
from src.utils.logging.callback_builder import CallbackBuilder
from src.utils.logging.callbacks.config_logger import ConfigLogger
from src.utils.logging.callbacks.csv_logger import CsvLogger
from src.utils.logging.callbacks.csv_plotter import CsvPlotter
from src.utils.settings import Settings


class HeadlineDoc2VecModelBuilder(ModelBuilder):
    MODEL_IDENTIFIER = 'headline_doc2vec_model'

    def __init__(self):
        super().__init__()

        self.required_inputs.append('headline_doc2vec')

        self.default_parameters['relu_fully_connected_dimensions'] = 256
        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'
        self.default_parameters['main_output'] = 'main_output'

    def __call__(self):
        super().prepare_building()

        headline_doc2vec = self.inputs['headline_doc2vec']

        headline_input = Input(shape=(headline_doc2vec.get_dimensions(),), name='headline_input')

        relu_fully_connected = Dense(self.parameters['relu_fully_connected_dimensions'], activation='relu')(
            headline_input)
        batch_normalization = BatchNormalization()(relu_fully_connected)
        main_output = Dense(1, activation='sigmoid', name=self.parameters['main_output'])(batch_normalization)

        model = Model(inputs=[headline_input], outputs=[main_output], name=self.MODEL_IDENTIFIER)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1])

        model.summary()
        return model


class HeadlineDoc2VecPreprocessor(Preprocessor):
    def __init__(self, model, headline_doc2vec):
        super().__init__(model)
        self.headline_doc2vec = headline_doc2vec

    def array_to_dict(self, data):
        result = {}
        headlines = []
        is_top_submission = []

        output_names = [l.name for l in self.model.output_layers]

        for article in data:
            headlines.append(self.headline_doc2vec.get_vector(article[LabelsView.HEADLINE.value]))
            is_top_submission.append(1 if article[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

        result['headlines'] = np.array(headlines)
        result['is_top_submission'] = np.array(is_top_submission, dtype=int)
        result['class_weights'] = self.calculate_class_weights(result['is_top_submission'], output_names)

        return result


def train():
    settings = Settings()
    default_parameters = settings.get_training_parameter_default()
    choices_parameters = settings.get_training_parameter_choices()

    arg_parse = ArgumentParser()
    arg_parse.add_argument('--batch_size', type=int, default=default_parameters['batch_size'])
    arg_parse.add_argument('--epochs', type=int, default=default_parameters['epochs'])
    arg_parse.add_argument('--h_doc2vec_dim', type=int, default=default_parameters['headline_doc2vec_dimensions'],
                           choices=choices_parameters['headline_doc2vec_dimensions'])

    arguments = arg_parse.parse_args()

    headline_doc2vec = Doc2Vec()
    headline_doc2vec.load_model('article', arguments.h_doc2vec_dim)

    model_builder = HeadlineDoc2VecModelBuilder() \
        .set_input('headline_doc2vec', headline_doc2vec)
    model = model_builder()

    preprocessor = HeadlineDoc2VecPreprocessor(model, headline_doc2vec)
    preprocessor.load_data()

    callbacks = CallbackBuilder(model, model_builder.default_parameters, arguments, [CsvLogger, CsvPlotter, ConfigLogger])()

    training_input = [preprocessor.training_data['headlines']]
    training_output = [preprocessor.training_data['is_top_submission']]

    validation_input = [preprocessor.validation_data['headlines']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = preprocessor.training_data['class_weights']

    model.fit(training_input, training_output, batch_size=arguments.batch_size,
              epochs=arguments.epochs, callbacks=callbacks,
              validation_data=(validation_input, validation_output), class_weight=class_weights)
