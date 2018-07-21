from argparse import ArgumentParser

import numpy as np
from keras.models import load_model

from src.encoder.glove import Glove
from src.encoder.numeric_log import NumericLog
from src.models.preprocessor import Preprocessor
from src.utils.f1_score import precision, recall, f1
from src.utils.settings import Settings


def calculate_correlations():
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--model_1', type=str)
    arg_parse.add_argument('--model_2', type=str)
    arg_parse.add_argument('--model_3', type=str)
    arg_parse.add_argument('--model_4', type=str)
    arg_parse.add_argument('--model_5', type=str)
    arg_parse.add_argument('--model_6', type=str)
    arg_parse.add_argument('--model_7', type=str)
    arguments = arg_parse.parse_args()

    settings = Settings()
    default_parameters = settings.get_training_parameter_default()

    glove = Glove(default_parameters['dictionary_size'])
    glove.load_embedding()

    headline_numeric_log = NumericLog(default_parameters['max_headline_length'])
    article_numeric_log = NumericLog(default_parameters['max_article_length'])

    print('load data...')
    preprocessor = Preprocessor(None)
    preprocessor.set_encoder('glove', glove)
    preprocessor.set_encoder('headline_numeric_log', headline_numeric_log)
    preprocessor.set_encoder('article_numeric_log', article_numeric_log)
    preprocessor.set_parameter('max_headline_length', default_parameters['max_headline_length'])
    preprocessor.set_parameter('body_begin_length', default_parameters['body_begin_length'])

    preprocessor.load_data(['headline',
                            'body_begin',
                            'category',
                            'minute',
                            'hour',
                            'day_of_week',
                            'day_of_year',
                            'headline_log_representation',
                            'article_log_representation',
                            'competitive_score'])

    custom_objects = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    print('load models...')
    model_inputs = {}
    model_inputs['model_1'] = [preprocessor.test_data['headline']]
    model_inputs['model_2'] = [preprocessor.test_data['headline']]
    model_inputs['model_3'] = [preprocessor.test_data['body_begin']]
    model_inputs['model_4'] = [preprocessor.test_data['category']]
    model_inputs['model_5'] = [preprocessor.test_data[key] for key in ['minute', 'hour', 'day_of_week', 'day_of_year']]
    model_inputs['model_6'] = [preprocessor.test_data[key] for key in
                               ['headline_log_representation', 'article_log_representation']]
    model_inputs['model_7'] = [preprocessor.test_data['competitive_score']]

    print('predict...')
    predictions = {}
    for model_name in model_inputs.keys():
        if hasattr(arguments, model_name) and getattr(arguments, model_name):
            model = load_model(getattr(arguments, model_name), custom_objects=custom_objects)
            predictions[model_name] = np.round(model.predict(model_inputs[model_name]))

    print('calculate correlation...')
    for model_name_1 in predictions.keys():
        for model_name_2 in predictions.keys():
            if model_name_1 != model_name_2:
                correlation = np.corrcoef(predictions[model_name_1][:, -1], predictions[model_name_2][:, -1])[0]
                print(model_name_1, model_name_2, correlation[1])
