import numpy as np
from keras import Input, Model
from keras.callbacks import CSVLogger
from keras.layers import Embedding, Dense, Reshape, concatenate, BatchNormalization

from src.data_handler.db_fields import LabelsView
from src.models.doc2vec import Doc2Vec
from src.prediction.model_builder import ModelBuilder
from src.prediction.preprocessor import Preprocessor
import src.utils.f1_score
from src.utils.csv_plot import CSVPlotterCallback
from src.utils.settings import Settings


class Doc2VecModelBuilder(ModelBuilder):

    MODEL_IDENTIFIER = 'doc2vec_model'

    def __init__(self):
        super().__init__()

        self.required_inputs.append('headline_doc2vec')

        self.default_parameters['hour_embedding_dimensions'] = 5
        self.default_parameters['minute_embedding_dimensions'] = 5
        self.default_parameters['day_of_week_embedding_dimensions'] = 5
        self.default_parameters['day_of_year_embedding_dimensions'] = 5
        self.default_parameters['relu_fully_connected_dimensions'] = 256
        self.default_parameters['optimizer'] = 'adam'
        self.default_parameters['loss'] = 'binary_crossentropy'
        self.default_parameters['main_output'] = 'main_output'

    def __call__(self):
        super().prepare_building()

        headline_doc2vec = self.inputs['headline_doc2vec']

        headline_input = Input(shape=(headline_doc2vec.get_dimensions(),), name='headline_input')

        hour_input = Input(shape=(1,), name='hour_input')
        hour_embedding = Embedding(24, self.parameters['hour_embedding_dimensions'])(hour_input)
        hour_reshape = Reshape((self.parameters['hour_embedding_dimensions'],))(hour_embedding)

        minute_input = Input(shape=(1,), name='minute_input')
        minute_embedding = Embedding(60, self.parameters['minute_embedding_dimensions'])(minute_input)
        minute_reshape = Reshape((self.parameters['minute_embedding_dimensions'],))(minute_embedding)

        day_of_week_input = Input(shape=(1,), name='day_of_week_input')
        day_of_week_embedding = Embedding(60, self.parameters['day_of_week_embedding_dimensions'])(day_of_week_input)
        day_of_week_reshape = Reshape((self.parameters['day_of_week_embedding_dimensions'],))(day_of_week_embedding)

        day_of_year_input = Input(shape=(1,), name='day_of_year_input')
        day_of_year_embedding = Embedding(366, self.parameters['day_of_year_embedding_dimensions'])(day_of_year_input)
        day_of_year_reshape = Reshape((self.parameters['day_of_year_embedding_dimensions'],))(day_of_year_embedding)

        embedding_concatenation = concatenate([headline_input,
                                               hour_reshape,
                                               minute_reshape,
                                               day_of_week_reshape,
                                               day_of_year_reshape])
        relu_fully_connected = Dense(self.parameters['relu_fully_connected_dimensions'], activation='relu')(
            embedding_concatenation)
        batch_normalization = BatchNormalization()(relu_fully_connected)
        main_output = Dense(1, activation='sigmoid', name=self.parameters['main_output'])(batch_normalization)

        model = Model(inputs=[headline_input,
                              hour_input,
                              minute_input,
                              day_of_week_input,
                              day_of_year_input, ], outputs=[main_output], name=self.MODEL_IDENTIFIER)

        model.compile(loss=self.parameters['loss'],
                      optimizer=self.parameters['optimizer'],
                      metrics=['accuracy', src.utils.f1_score.precision, src.utils.f1_score.recall, src.utils.f1_score.f1])

        model.summary()
        return model


class Doc2VecPreprocessor(Preprocessor):
    def __init__(self, model, headline_doc2vec):
        super().__init__(model)
        self.headline_doc2vec = headline_doc2vec

    def array_to_dict(self, data):
        result = {}
        headlines = []
        hours = []
        minutes = []
        day_of_weeks = []
        day_of_years = []
        is_top_submission = []

        output_names = [l.name for l in self.model.output_layers]

        for article in data:
            headlines.append(self.headline_doc2vec.get_vector(article[LabelsView.HEADLINE.value]))
            hours.append(article[LabelsView.HOUR.value])
            minutes.append(article[LabelsView.MINUTE.value])
            day_of_weeks.append(article[LabelsView.DAY_OF_WEEK.value] - 1)
            day_of_years.append(article[LabelsView.DAY_OF_YEAR.value] - 1)
            is_top_submission.append(1 if article[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

        result['headlines'] = np.array(headlines)
        result['hours'] = np.array(hours, dtype=int)
        result['minutes'] = np.array(minutes, dtype=int)
        result['day_of_weeks'] = np.array(day_of_weeks, dtype=int)
        result['day_of_years'] = np.array(day_of_years, dtype=int)
        result['is_top_submission'] = np.array(is_top_submission, dtype=int)
        result['class_weights'] = self.calculate_class_weights(result['is_top_submission'], output_names)

        return result


def train():
    settings = Settings()

    batch_size = 64
    epochs = 20

    headline_doc2vec = Doc2Vec()
    headline_doc2vec.load_model('headline', 100)

    model_builder = Doc2VecModelBuilder().set_input('headline_doc2vec', headline_doc2vec)
    model = model_builder()

    preprocessor = Doc2VecPreprocessor(model, headline_doc2vec)
    preprocessor.load_data()

    csv_filename = settings.get_csv_filename(model.name)

    csv_logger = CSVLogger(csv_filename)

    plot_config = [('f1', (0.1, 0.0, 0.9), 'f1-score'), ('val_f1', 'g', 'validation f1-score')]
    plot_callback = CSVPlotterCallback(csv_filename, plot_config)

    training_input = [preprocessor.training_data['headlines'],
                      preprocessor.training_data['hours'],
                      preprocessor.training_data['minutes'],
                      preprocessor.training_data['day_of_weeks'],
                      preprocessor.training_data['day_of_years']]

    training_output = [preprocessor.training_data['is_top_submission']]

    validation_input = [preprocessor.validation_data['headlines'],
                        preprocessor.validation_data['hours'],
                        preprocessor.validation_data['minutes'],
                        preprocessor.validation_data['day_of_weeks'],
                        preprocessor.validation_data['day_of_years']]

    validation_output = [preprocessor.validation_data['is_top_submission']]

    class_weights = preprocessor.training_data['class_weights']

    model.fit(training_input, training_output, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger, plot_callback],
              validation_data=(validation_input, validation_output), class_weight=class_weights)
