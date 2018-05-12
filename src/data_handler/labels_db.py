from src.data_handler.db_fields import Articles, Labels, Comments
from src.data_handler.postgres_db import PostgresDb


class LabelsDb(PostgresDb):

    def get_comment_count(self, article_id):
        try:
            table = Comments.__name__.lower()

            select = 'SELECT COUNT(*) FROM {0} WHERE {1}={2}'.format(table, Comments.ID.value, article_id)
            self.cur.execute(select)
            result = self.cur.fetchone()
            return result['COUNT']
        except Exception as e:
            print("cant get comment count")
            print(e)

    def get_article(self, article_id):
        return self.get_dict(Labels, article_id)

    def get_articles(self):
        return self.get_dicts(Labels)

    def save_labels(self, labels_dict):
        self.insert_dict(Labels, labels_dict)
