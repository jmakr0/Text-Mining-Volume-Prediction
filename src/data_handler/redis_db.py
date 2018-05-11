import json

import redis as redis

from src.utils.settings import Settings


class RedisDb:
    def __init__(self, database_name):
        settings = Settings()
        database_settings = settings.get_database_settings(database_name)

        self.r = redis.StrictRedis(**database_settings)

    def get(self, key):
        return json.loads(self.r.get(key))

    def set(self, key, value):
        self.r.set(key, json.dumps(value))

    def __len__(self):
        return self.r.dbsize()

    def __iter__(self):
        for key in self.r.scan_iter():
            value = self.r.get(key)
            yield key, value
