from threading import Semaphore, Thread

from src.data_handler.download_errors_db import DownloadErrorsDb
from src.data_handler.guardian_article_downloads_db import GuardianArticleDownloadsDb
from src.data_handler.guardian_csv import GuardianCsvData
from src.guardian.api import Api

sem = Semaphore()


class Downloader:
    def __init__(self, worker_count, urls):
        self.urls = []
        if type(urls) is list:
            for url in urls:
                self.urls.append((url[0], url[1]))
        elif type(urls) is dict:
            for key, url in urls.items():
                self.urls.append((key, url))

        self.workers = []
        for i in range(worker_count):
            self.workers.append(DownloadWorker(self.get_url))

    def get_url(self):
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
    def __init__(self, get_url):
        super().__init__()

        self.download_db = GuardianArticleDownloadsDb()
        self.error_db = DownloadErrorsDb()

        self.get_url = get_url
        self.guardian_api = Api()

    def run(self):
        run = True
        api_key = self.guardian_api.get_api_key()

        while run:
            url_tuple = self.get_url()

            if url_tuple:
                url_saved = self.download_db.get(url_tuple[0])
                url_error = self.error_db.get(url_tuple[0])
                try:
                    if not url_saved and not url_error:
                        response = self.guardian_api.request_article(url_tuple[1], api_key)

                        has_error = self.guardian_api.has_error(response)
                        has_rate_limit = self.guardian_api.rate_limit(response)

                        if not has_error and not has_rate_limit:
                            article = self.guardian_api.extract_fields(response)
                            self.download_db.set(url_tuple[0], article)
                        elif has_error and not has_rate_limit:
                            self.error_db.set(url_tuple[0], url_tuple[1])
                        elif has_rate_limit:
                            api_key = self.guardian_api.get_api_key()
                            run = not not api_key

                except Exception as e:
                    print('Could not get: ' + url_tuple[1])
                    print(str(response))
            else:
                run = False


def download():
    urls = GuardianCsvData().get_url_tuples()
    # max 12 requests per second
    worker_count = 5

    scraper = Downloader(worker_count, urls)
    scraper.start()
