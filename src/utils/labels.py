from enum import Enum


class Labels(Enum):
    HEADLINE = 'headline'
    COMMENT_CLOSE_DATE = 'comment_close_date'
    STANDFIRST = 'standfirst'
    COMMENTABLE = 'commentable'
    IS_PREMODERATED = 'is_premoderated'
    LAST_MODIFIED = 'last_modified'
    NEWSPAPER_EDITION_DATE = 'newspaper_edition_date'
    LEGALLY_SENSITIVE = 'legally_sensitive'
    ARTICLE = 'article'

    HEADLINE_WORD_COUNT = 'headline_word_count'
    STANDFIRST_WORD_COUNT = 'standfirst_word_count'
    ARTICLE_WORD_COUNT = 'article_word_count'
    COMMENT_PERFORMANCE_WEEKLY = 'comment_performance_weekly'
    IN_TOP_TEN_PERCENT = 'in_top_ten_percent'
    COMMENT_NUMBER = 'comment_number'
    UNIX_TIMESTAMP = 'unix_timestamp'
    DAY_OF_WEEK = 'day_of_week'
    DAY_OF_YEAR = 'day_of_year'
    HOUR = 'hour'
    MINUTE = 'minute'
    GENRE = 'genre'
