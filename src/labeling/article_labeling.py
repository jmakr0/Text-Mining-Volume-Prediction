from src.data_handler.guardian_api_db import GuardianApiDb
from src.data_handler.labels_db import LabelsDb
from src.model.article import Article


class ArticleLabeling:

    def __init__(self):
        self.download_db = GuardianApiDb()
        self.labels_db = LabelsDb()

    def start(self):
        for download in iter(self.download_db.get_articles()):
            article_labeled = Article(download, self.labels_db)

            self.labels_db.save_labels(article_labeled.get_labels())
