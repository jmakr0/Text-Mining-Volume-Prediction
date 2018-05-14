from enum import Enum


class Labels(Enum):
    # Guardian API
    HEADLINE = 'headline'
    COMMENT_CLOSE_DATE = 'commentCloseDate'
    STANDFIRST = 'standfirst'
    COMMENTABLE = 'commentable'
    IS_PREMODERATED = 'isPremoderated'
    LAST_MODIFIED = 'lastModified'
    NEWSPAPER_EDITION_DATE = 'newspaperEditionDate'
    LEGALLY_SENSITIVE = 'legallySensitive'
    ARTICLE = 'bodyText'
    GENRE_ID = 'sectionId'
    GENRE_NAME = 'sectionName'
    PUBLICATION_DATE = 'webPublicationDate'
    # Own fields
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
