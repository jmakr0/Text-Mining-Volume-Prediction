from src.data_handler.articles_db import ArticlesDb
from src.data_handler.guardian_api_db import GuardianApiDb
from src.model.article import Article


class ArticleLabeling:
    def __init__(self):
        self.download_db = GuardianApiDb()
        self.article_db = ArticlesDb()

    def start(self):
        for article_id, article in iter(self.download_db):
            article_id = int(article_id)
            article = Article(article_id, article)

            self.article_db.set(article_id, article.article_dict)
