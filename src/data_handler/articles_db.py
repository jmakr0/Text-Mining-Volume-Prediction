from src.data_handler.db_fields import Articles, GuardianApi, GuardianApiError
from src.data_handler.postgres_db import PostgresDb


class ArticlesDb(PostgresDb):

    def get_download_articles(self):
        try:
            select = 'SELECT * FROM downloadview'
            self.cur.execute(select)
            return self.cur.fetchall()
        except Exception as e:
            print("cant get dicts")
            print(e)
