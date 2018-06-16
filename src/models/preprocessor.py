import random

import numpy as np

from src.data_handler.labels_db import LabelsDb

AUX_OUTPUT_NAME = 'aux_out'
MAIN_OUTPUT_NAME = 'main_out'


class Preprocessor:

    def __init__(self, model):
        self.model = model
        self.db = LabelsDb()

        self.training_data = {}
        self.validation_data = {}
        self.test_data = {}

    @staticmethod
    def split_data(data, seed=42, training_size=0.7, validation_size=0.15, test_size=0.15):
        assert (training_size + validation_size + test_size) == 1
        # shuffle the same every time
        random.seed(seed)
        random.shuffle(data)

        num_tuples = len(data)
        num_train_tuples = int(num_tuples * training_size)
        num_validation_tuples = int(num_tuples * validation_size)

        training_data = data[:num_train_tuples]
        validation_data = data[num_train_tuples:num_train_tuples + num_validation_tuples]
        test_data = data[num_train_tuples + num_validation_tuples:]

        return training_data, validation_data, test_data

    @staticmethod
    def calculate_class_weights(is_top_submission, output_names):
        classes, counts = np.unique(is_top_submission, return_counts=True)
        class_weight_dict = dict(zip(classes, counts))

        return {name: class_weight_dict for name in output_names}

    def load_data(self):
        labeled_articles = self.db.get_labeled_data()

        training_data, validation_data, test_data = self.split_data(labeled_articles)

        self.training_data = self.array_to_dict(training_data)
        self.validation_data = self.array_to_dict(validation_data)
        self.test_data = self.array_to_dict(test_data)

    def array_to_dict(self, data):
        """
        This methods has to be implemented in specific child class.
        Transforms data array to dict input arrays.
        :param data: data array
        :return: dict
        """
        raise NotImplementedError
