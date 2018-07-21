from enum import Enum

from src.data_handler.db_fields import LabelsView
from src.data_handler.postgres_db import PostgresDb


class LabelsDb(PostgresDb):

    def get_labeled_data(self):
        return self.get_dicts(LabelsView_Small, LabelsView_Small.UNIX_TIMESTAMP.value + ' ASC')

class LabelsView_Small(Enum):
    ID = 'article_id'
    HEADLINE = 'headline'
    HEADLINE_WORD_COUNT = 'headline_word_count'
    ARTICLE = 'article'
    ARTICLE_WORD_COUNT = 'article_word_count'
    CATEGORY_ID = 'category_id'
    COMMENT_PERFORMANCE_WEEKLY = 'comment_performance_weekly'
    IN_TOP_TEN_PERCENT = 'in_top_ten_percent'
    COMMENT_COUNT = 'comment_count'
    UNIX_TIMESTAMP = 'unix_timestamp'
    DAY_OF_WEEK = 'day_of_week'
    DAY_OF_YEAR = 'day_of_year'
    HOUR = 'hour'
    MINUTE = 'minute'
    COMPETITIVE_SCORE = 'competitive_score'