import json

from src.utils.labels import Labels


class Article:
    def __init__(self, article_id, article_bytes):
        self.article_id = article_id
        self.article_dict = json.loads(article_bytes.decode('utf-8').replace('\'', '"'))

    @property
    def headline(self):
        return self.article_dict[Labels.HEADLINE.value]
