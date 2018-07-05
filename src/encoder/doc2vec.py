import os
import random
import logging

from gensim.models import doc2vec
from gensim.utils import simple_preprocess

from src.data_handler.db_fields import LabelsView
from src.data_handler.labels_db import LabelsDb
from src.utils.settings import Settings

# Set up log to terminal
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Doc2Vec:
    def __init__(self):
        self.db = LabelsDb()
        self.model = None

    def load_model(self, tag, dimensions):
        doc2vec_dir = Settings().get_doc2vec_dir()
        doc2vec_file = '{}/{}_{}.model'.format(doc2vec_dir, tag, dimensions)

        if os.path.isfile(doc2vec_file):
            print('loading doc2vec model ...')
            self.model = doc2vec.Doc2Vec.load(doc2vec_file)
        else:
            self._train_model(tag, dimensions)
            print('saving model ...')
            try:
                os.makedirs(doc2vec_dir)
            except FileExistsError:
                pass
            self.model.save(doc2vec_file)

    def _train_model(self, tag, dimensions):
        """
        :param tag: Defines on which dataset to train. Choose either 'headline' or 'text'.
        :param dimensions:
        :return:
        """
        print('training model ...')
        if tag == 'headline':
            column = LabelsView.HEADLINE.value
        elif tag == 'article':
            column = LabelsView.ARTICLE.value
        else:
            raise ValueError('tag is not accepted')

        docs = self._load_docs(column)
        self.model = doc2vec.Doc2Vec(vector_size=dimensions, min_count=5, epochs=50, workers=18)
        self.model.build_vocab(docs)
        self.model.train(docs, total_examples=self.model.corpus_count, epochs=self.model.epochs, report_delay=10)

    def _load_docs(self, column):
        data = self.db.get_labeled_data()
        docs = []

        for i, row in enumerate(data):
            value = row[column]
            if value:
                docs.append(doc2vec.TaggedDocument(simple_preprocess(value), [i]))

        random.seed(187)
        random.shuffle(docs)

        return docs

    def get_dimensions(self):
        return self.model.vector_size

    def get_vector(self, doc):
        return self.model.infer_vector(simple_preprocess(doc))
