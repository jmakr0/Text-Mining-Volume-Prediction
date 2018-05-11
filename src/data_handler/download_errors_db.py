from src.data_handler.redis_db import RedisDb


class DownloadErrorsDb(RedisDb):
    def __init__(self):
        database_name = 'download-errors'
        super().__init__(database_name)
