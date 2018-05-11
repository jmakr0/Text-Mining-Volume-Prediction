import pandas

from src.utils.settings import Settings


class GuardianCsv:
    def __init__(self):
        settings = Settings()

        self.articles_filename = settings.get_csv_file('articles')
        self.authors_filename = settings.get_csv_file('authors')
        self.comments_filename = settings.get_csv_file('comments')

        self._articles_df = None
        self._authors_df = None
        self._comments_df = None

    def _get_articles_df(self):
        if not self._articles_df:
            self._articles_df = pandas.read_csv(self.articles_filename)
        return self._articles_df

    def _get_authors_df(self):
        if not self._authors_df:
            self._authors_df = pandas.read_csv(self.authors_filename)
        return self._authors_df

    def _get_comments_df(self):
        if not self._comments_df:
            self._comments_df = pandas.read_csv(self.comments_filename)
        return self._comments_df

    def get_url_tuples(self):
        return self._get_articles_df().values.tolist()
