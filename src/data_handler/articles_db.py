from src.data_handler.db_fields import Articles
from src.data_handler.postgres_db import PostgresDb


class ArticlesDb(PostgresDb):

    def get_articles(self):
        return self.get_dicts(Articles)
