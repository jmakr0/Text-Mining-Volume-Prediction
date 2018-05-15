import csv

from src.data_handler.db_fields import Comments, Articles, Authors
from src.data_handler.postgres_db import PostgresDb
from src.utils.settings import Settings


class GuardianCsvData:
    def __init__(self):
        settings = Settings()

        self.db = PostgresDb()
        self.articles_filename = settings.get_csv_file('articles')
        self.authors_filename = settings.get_csv_file('authors')
        self.comments_filename = settings.get_csv_file('comments')

    def _import(self, filename, db_fields):
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for comment_dict in reader:
                self.db.insert_dict(db_fields, comment_dict)

    def import_article(self):
        self._import(self.articles_filename, Articles)

    def import_authors(self):
        self._import(self.authors_filename, Authors)

    def import_comments(self):
        self._import(self.comments_filename, Comments)
