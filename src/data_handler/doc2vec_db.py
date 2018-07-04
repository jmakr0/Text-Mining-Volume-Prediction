from src.data_handler.db_fields import Doc2Vec
from src.data_handler.postgres_db import PostgresDb


class Doc2VecDb(PostgresDb):

    def save_vector(self, article_id, vector_list, dimensions, tag):
        doc = {Doc2Vec.ID.value: article_id, Doc2Vec.VECTOR.value: vector_list, Doc2Vec.DIMENSIONS.value: dimensions,
               Doc2Vec.TAG.value: tag}
        self.insert_dict(Doc2Vec, doc)
