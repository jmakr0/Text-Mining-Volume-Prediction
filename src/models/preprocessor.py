import random

import numpy as np
from keras.preprocessing import sequence

from src.data_handler.db_fields import LabelsView
from src.data_handler.labels_db import LabelsDb


class Preprocessor:
    def __init__(self, model, training_size=0.7, validation_size=0.15, test_size=0.15, split_seed=42):
        self.model = model
        self.training_size = training_size
        self.validation_size = validation_size
        self.test_size = test_size
        assert self.training_size + self.validation_size + self.test_size == 1
        self.split_seed = split_seed

        self.encoders = {}
        self.parameters = {}

        self.db = LabelsDb()

        self.training_data = {}
        self.validation_data = {}
        self.test_data = {}

    def set_encoder(self, name, encoder):
        self.encoders[name] = encoder

    def get_encoder(self, name):
        if name not in self.encoders:
            raise ValueError('Missing required encoder {}'.format(name))
        else:
            return self.encoders[name]

    def set_parameter(self, name, parameter):
        self.parameters[name] = parameter

    def get_parameter(self, name):
        if name not in self.parameters:
            raise ValueError('Missing required parameter {}'.format(name))
        else:
            return self.parameters[name]

    def load_data(self, data_fields):
        """
        :param data_fields: The names of the network inputs.
        :param network_outputs: The names of the network outputs.
        """
        labeled_articles = self.db.get_labeled_data()
        training_data, validation_data, test_data = self.split_data(labeled_articles)
        self.training_data = self.array_to_dict(training_data, data_fields)
        self.validation_data = self.array_to_dict(validation_data, data_fields)
        self.test_data = self.array_to_dict(test_data, data_fields)

    def array_to_dict(self, data, data_fields):
        extractors = {}
        for data_field in data_fields:
            extractors[data_field] = getattr(self, '_{}_extractor'.format(data_field))

        result = {key: [] for key in extractors}
        for row in data:
            for key, extractor in extractors.items():
                result[key].append(extractor(row))

        for key in result.keys():
            processor = getattr(self, '_{}_processor'.format(key))
            if processor:
                result[key] = processor(result[key])

        return result

    def _headline_extractor(self, row):
        return self.get_encoder('glove').text_to_sequence(row[LabelsView.HEADLINE.value])

    def _headline_processor(self, headlines):
        return sequence.pad_sequences(headlines, maxlen=self.get_parameter('max_headline_length'))

    def _is_top_submission_extractor(self, row):
        return 1 if row[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0

    def _is_top_submission_processor(self, top_submissions):
        return np.array(top_submissions, dtype=int)

    def _body_begin_extractor(self, row):
        return self.get_encoder('glove').text_to_sequence(row[LabelsView.ARTICLE.value],
                                                          limit=self.get_parameter('body_begin_length'))

    def _body_begin_processor(self, body_begins):
        return sequence.pad_sequences(body_begins, maxlen=self.get_parameter('body_begin_length'))

    def _article_vector_extractor(self, row):
        return self.get_encoder('article_doc2vec').get_vector(row[LabelsView.ARTICLE.value])

    def _article_vector_processor(self, article_vectors):
        return np.array(article_vectors)

    def _category_extractor(self, row):
        return row[LabelsView.CATEGORY_ID.value]

    def _category_processor(self, categories):
        return np.array(categories, dtype=int)

    def _minute_extractor(self, row):
        return row[LabelsView.MINUTE.value]

    def _minute_processor(self, minutes):
        return np.array(minutes, dtype=int)

    def _hour_extractor(self, row):
        return row[LabelsView.HOUR.value]

    def _hour_processor(self, hours):
        return np.array(hours, dtype=int)

    def _day_of_week_extractor(self, row):
        return row[LabelsView.DAY_OF_WEEK.value]

    def _day_of_week_processor(self, days_of_weeks):
        return np.array(days_of_weeks, dtype=int)

    def _day_of_year_extractor(self, row):
        return row[LabelsView.DAY_OF_YEAR.value] - 1

    def _day_of_year_processor(self, days_of_years):
        return np.array(days_of_years, dtype=int)

    def _headline_log_representation_extractor(self, row):
        return self.get_encoder('headline_numeric_log')(row[LabelsView.HEADLINE_WORD_COUNT.value])

    def _headline_log_representation_processor(self, headline_log_representations):
        return np.array(headline_log_representations, dtype=int)

    def _article_log_representation_extractor(self, row):
        return self.get_encoder('article_numeric_log')(row[LabelsView.ARTICLE_WORD_COUNT.value])

    def _article_log_representation_processor(self, article_log_representation):
        return np.array(article_log_representation, dtype=int)

    def _competitive_score_extractor(self, row):
        return row[LabelsView.COMPETITIVE_SCORE.value]

    def _competitive_score_processor(self, competitive_scores):
        return np.array(competitive_scores)

    def split_data(self, data, shuffle=True):
        num_tuples = len(data)
        num_train_tuples = int(num_tuples * self.training_size)
        num_validation_tuples = int(num_tuples * self.validation_size)

        training_data = data[:num_train_tuples]
        validation_data = data[num_train_tuples:num_train_tuples + num_validation_tuples]
        test_data = data[num_train_tuples + num_validation_tuples:]

        # shuffle data
        if shuffle:
            random.seed(self.split_seed)
            random.shuffle(training_data)

        return training_data, validation_data, test_data
