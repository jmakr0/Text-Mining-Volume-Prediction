import json
from threading import Semaphore, Thread

from src.data_handler.articles_db import ArticlesDb
from src.data_handler.db_fields import GuardianApiError
from src.data_handler.download_errors_db import DownloadErrorsDb
from src.data_handler.guardian_api_db import GuardianApiDb
from src.guardian.api import Api

sem = Semaphore()


class Downloader:
    def __init__(self, worker_count, articles):
        self.urls = []
        if type(articles) is list:
            for article in articles:
                self.urls.append((article[0], article[1]))

        self.workers = []
        for i in range(worker_count):
            self.workers.append(DownloadWorker(self.get_article))

    def get_article(self):
        sem.acquire()
        url = False
        if len(self.urls) > 0:
            url = self.urls.pop(0)
        sem.release()
        return url

    def start(self):
        for worker in self.workers:
            worker.start()
        for worker in self.workers:
            worker.join()


class DownloadWorker(Thread):
    def __init__(self, get_article):
        super().__init__()

        self.download_db = GuardianApiDb()
        self.error_db = DownloadErrorsDb()

        self.get_article = get_article
        self.guardian_api = Api()

    def run(self):
        run = True
        api_key = self.guardian_api.get_api_key()

        while run:
            article = self.get_article()

            if article:
                id = article[0]
                url = article[1]
                try:
                    response = self.guardian_api.request_article(url, api_key)

                    has_error = self.guardian_api.has_error(response)
                    has_rate_limit = self.guardian_api.rate_limit(response)

                    if not has_error and not has_rate_limit:
                        article_dict = self.guardian_api.extract_fields(response)
                        article_dict[GuardianApiError.ID.value] = id
                        self.download_db.save_article(article_dict)
                    elif has_error and not has_rate_limit:
                        article_dict = {GuardianApiError.ID.value: id,
                                        GuardianApiError.REASON.value: json.dumps(response)}
                        self.error_db.save_article(article_dict)
                    elif has_rate_limit:
                        api_key = self.guardian_api.get_api_key()
                        run = not not api_key

                except Exception as e:
                    print('Could not get: ' + url)
                    print(str(response))
            else:
                run = False


def download():
    article_db = ArticlesDb()
    articles = article_db.get_download_articles()
    # max 12 requests per second
    worker_count = 8

    scraper = Downloader(worker_count, articles)
    scraper.start()
