from src.data_handler.db_fields import LabelsView
from src.data_handler.doc2vec_db import Doc2VecDb
from src.encoder.doc2vec import Doc2Vec


class CompetitiveScore:
    def __init__(self):
        # todo: use parameter
        self.dimensions = 50
        self.tag = 'headline'
        self.doc2Vec = Doc2Vec()
        self.doc2Vec.load_model(self.tag, self.dimensions);

    def save_doc2vec(self):
        doc2vec_db = Doc2VecDb()
        data = self.doc2Vec.db.get_labeled_data()

        for i, row in enumerate(data):
            article_id = row[LabelsView.ID.value];
            article = row[LabelsView.ARTICLE.value];
            vector = self.doc2Vec.get_vector(article)
            doc2vec_db.save_vector(article_id, vector.tolist(), self.dimensions, self.tag)

