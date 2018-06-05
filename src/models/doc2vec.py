import random
import time

import gensim as gensim
import numpy as np

from src.data_handler.db_fields import LabelsView
from src.data_handler.labels_db import LabelsDb
from src.utils.settings import Settings


class Doc2Vec:
    DIMENSIONS = 50

    def __init__(self):
        self.db = LabelsDb()
        self.test_docs = []
        self.training_docs = []

        self.read_corpus()
        self.train_model()

    def read_corpus(self):
        data = self.db.get_labeled_data()

        random.seed(187)
        random.shuffle(data)

        test_data = data[:int(len(data) * 0.1)]
        training_data = data[int(len(data) * 0.1):]

        for row in test_data:
            value = row[LabelsView.HEADLINE.value]
            self.test_docs.append(gensim.utils.simple_preprocess(value))

        for i, row in enumerate(training_data):
            value = row[LabelsView.HEADLINE.value]
            self.training_docs.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(value), [i]))

        # print(self.test_docs[:100])

    def train_model(self):
        model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=5, epochs=55, workers=4)
        model.build_vocab(self.training_docs)
        model.train(self.training_docs, total_examples=model.corpus_count, epochs=model.epochs)

        vec_1 = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
        vec_2 = model.infer_vector(['whiplash', 'testament', 'of', 'youth', 'wild', 'this', 'week', 'new', 'films'])
        vec_3 = model.infer_vector(['elisabeth', 'leonskaja', 'and', 'friends', 'birthday', 'tribute', 'to', 'virtuoso', 'pianist'])
        vec_4 = model.infer_vector(['elisabeth', 'leonskaja', 'or', 'friends', 'birthday', 'tribute', 'to', 'virtuoso', 'pianist'])

        print(np.linalg.norm(vec_1 - vec_2))
        print(np.linalg.norm(vec_3 - vec_4))
