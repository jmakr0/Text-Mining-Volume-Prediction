from src.data_handler.redis_db import RedisDb


class GuardianArticleDownloadsDb(RedisDb):
    def __init__(self):
        database_name = 'guardian-article-downloads'
        super().__init__(database_name)
