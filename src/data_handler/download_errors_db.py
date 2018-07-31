from src.data_handler.db_fields import GuardianApiError
from src.data_handler.postgres_db import PostgresDb


class DownloadErrorsDb(PostgresDb):

    def get_article(self, article_id):
        return self.get_dict(GuardianApiError, article_id)

    def save_article(self, article_dict):
        self.insert_dict(GuardianApiError, article_dict)
