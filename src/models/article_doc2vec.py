import numpy as np
from keras import Input, Model
from keras.layers import Dense, BatchNormalization

from src.data_handler.db_fields import LabelsView
from src.encoder.doc2vec import Doc2Vec
from src.models.model_builder import ModelBuilder
from src.models.preprocessor import Preprocessor
from src.utils.f1_score import precision, recall, f1
from src.utils.logging.callback_builder import CallbackBuilder
from src.utils.logging.callbacks.config_logger import ConfigLogger
from src.utils.logging.callbacks.csv_logger import CsvLogger
from src.utils.logging.callbacks.csv_plotter import CsvPlotter


class ArticleDoc2VecModelBuilder(ModelBuilder):
    MODEL_IDENTIFIER = 'article_doc2vec_model'

    def __init__(self):
        super().__init__()

        self.required_inputs.append('article_doc2vec')

        self.default_parameters['relu_fully_connected_dimensions'] = 256
        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'
        self.default_parameters['main_output'] = 'main_output'

    def __call__(self):
        super().prepare_building()

        article_doc2vec = self.inputs['article_doc2vec']

        article_input = Input(shape=(article_doc2vec.get_dimensions(),), name='article_input')

        relu_fully_connected = Dense(self.parameters['relu_fully_connected_dimensions'], activation='relu')(
            article_input)
        batch_normalization = BatchNormalization()(relu_fully_connected)
        main_output = Dense(1, activation='sigmoid', name=self.parameters['main_output'])(batch_normalization)

        model = Model(inputs=[article_input], outputs=[main_output], name=self.MODEL_IDENTIFIER)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', precision, recall, f1])

        model.summary()
        return model


class ArticleDoc2VecPreprocessor(Preprocessor):
    def __init__(self, model, article_doc2vec):
        super().__init__(model)
        self.article_doc2vec = article_doc2vec

    def array_to_dict(self, data):
        result = {}
        articles = []
        is_top_submission = []

        output_names = [l.name for l in self.model.output_layers]

        for row in data:
            articles.append(self.article_doc2vec.get_vector(row[LabelsView.ARTICLE.value]))
            is_top_submission.append(1 if row[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

        result['articles'] = np.array(articles)
        result['is_top_submission'] = np.array(is_top_submission, dtype=int)
        result['class_weights'] = self.calculate_class_weights(result['is_top_submission'], output_names)

        return result


def train():
    hyper_parameters = {}

    hyper_parameters['batch_size'] = 64
    hyper_parameters['epochs'] = 50

    article_doc2vec = Doc2Vec()
    article_doc2vec.load_model('article', 300)

    model_builder = ArticleDoc2VecModelBuilder() \
        .set_input('article_doc2vec', article_doc2vec)
    model = model_builder()

    preprocessor = ArticleDoc2VecPreprocessor(model, article_doc2vec)
    preprocessor.load_data()

    callbacks = CallbackBuilder(model, hyper_parameters, [CsvLogger, CsvPlotter, ConfigLogger])()

    training_input = [preprocessor.training_data['articles']]
    training_output = [preprocessor.training_data['is_top_submission']]

    validation_input = [preprocessor.validation_data['articles']]
    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = preprocessor.training_data['class_weights']
    model.fit(training_input, training_output, batch_size=hyper_parameters['batch_size'],
              epochs=hyper_parameters['epochs'], callbacks=callbacks,
              validation_data=(validation_input, validation_output), class_weight=class_weights)
