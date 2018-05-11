from src.data_handler.redis_db import RedisDb


class ArticlesDb(RedisDb):
    def __init__(self):
        database_name = 'articles'
        super().__init__(database_name)
