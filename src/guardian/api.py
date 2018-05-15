from urllib.parse import urlparse

import requests

from src.data_handler.db_fields import GuardianApi
from src.utils.settings import Settings


class Api:
    ENDPOINT = 'https://content.guardianapis.com'

    def __init__(self):
        settings = Settings()
        self.api_keys = settings.get_guardian_api_keys()
        self.fields = list(map(lambda e: e.value, GuardianApi))

    def _normalize_fields(self, fields):
        for field in GuardianApi:
            if field.value not in fields:
                fields[field.value] = None
        return fields

    def get_api_key(self):
        if len(self.api_keys) > 0:
            return self.api_keys.pop()

    def request_article(self, url, api_key):
        url = self.ENDPOINT + urlparse(url).path
        url += '?api-key=' + api_key
        url += '&show-fields=' + ','.join(self.fields)
        return requests.get(url).json()

    def extract_fields(self, response):
        content = response['response']['content']
        fields = content['fields']

        fields = self._normalize_fields(fields)

        fields[GuardianApi.GENRE_ID.value] = content[GuardianApi.GENRE_ID.value]
        fields[GuardianApi.GENRE_NAME.value] = content[GuardianApi.GENRE_NAME.value]
        fields[GuardianApi.PUBLICATION_DATE.value] = content[GuardianApi.PUBLICATION_DATE.value]
        return fields

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
