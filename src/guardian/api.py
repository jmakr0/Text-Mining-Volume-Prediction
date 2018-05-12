from urllib.parse import urlparse

import requests

from src.utils.settings import Settings


class Api:
    ENDPOINT = 'https://content.guardianapis.com'
    FIELDS = ['headline', 'standfirst', 'commentCloseDate', 'commentable', 'isPremoderated', 'lastModified', 'newspaperEditionDate', 'legallySensitive', 'bodyText']

    def __init__(self):
        settings = Settings()
        self.api_keys = settings.get_guardian_api_keys()

    def get_api_key(self):
        if len(self.api_keys) > 0:
            return self.api_keys.pop()

    def request_article(self, url, api_key):
        url = self.ENDPOINT + urlparse(url).path
        url += '?api-key=' + api_key
        url += '&show-fields=' + ','.join(self.FIELDS)
        return requests.get(url).json()

    @staticmethod
    def extract_fields(response):
        return response['response']['content']['fields']

    @staticmethod
    def has_error(response):
        if 'response' in response:
            return response['response']['status'] == 'error'
        else:
            return False

    @staticmethod
    def rate_limit(response):
        if 'message' in response:
            return response['message'] == 'API rate limit exceeded'
        else:
            return False
