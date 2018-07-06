from argparse import ArgumentParser

from keras.models import load_model

from src.encoder.glove import Glove
from src.evaluators.correlation_evaluator import CorrelationEvaluator
from src.models.headline import HeadlinePreprocessor
from src.models.lstm_body_begin import LstmBodyBeginPreprocessor
from src.utils.f1_score import precision, recall, f1
from src.utils.settings import Settings


def evaluate():
    settings = Settings()
    default_parameters = settings.get_training_parameter_default()

    arg_parse = ArgumentParser()
    arg_parse.add_argument('--headline_model', type=str, required=True)
    arg_parse.add_argument('--lstm_model', type=str, required=True)
    arg_parse.add_argument('--max_headline_length', type=int, default=default_parameters['max_headline_length'])
    arg_parse.add_argument('--body_begin_length', type=int, default=default_parameters['body_begin_length'])
    arg_parse.add_argument('--dictionary_size', type=int, default=default_parameters['dictionary_size'])

    arguments = arg_parse.parse_args()

    # necessary for self defined metrics in model
    custom_objects = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    headline_model = load_model(arguments.headline_model, custom_objects=custom_objects)
    lstm_model = load_model(arguments.lstm_model, custom_objects=custom_objects)

    glove = Glove(arguments.dictionary_size)
    glove.load_embedding()

    headline_preprocessor = HeadlinePreprocessor(headline_model, glove, arguments.max_headline_length)
    headline_preprocessor.load_data()
    lstm_preprocessor = LstmBodyBeginPreprocessor(lstm_model, glove, arguments.body_begin_length)
    lstm_preprocessor.load_data()

    evaluator = CorrelationEvaluator(headline_model, lstm_model, headline_preprocessor.test_data['headlines'],
                                     lstm_preprocessor.test_data['body_beginnings'])
    evaluator()
