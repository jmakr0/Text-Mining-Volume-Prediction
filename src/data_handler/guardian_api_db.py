from src.data_handler.db_fields import GuardianApi
from src.data_handler.postgres_db import PostgresDb


class GuardianApiDb(PostgresDb):

    def get_article(self, article_id):
        return self.get_dict(GuardianApi, article_id)

    def save_article(self, article_dict):
        self.insert_dict(GuardianApi, article_dict)
