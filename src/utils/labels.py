from enum import Enum


class Labels(Enum):
    # api fields
    HEADLINE = 'headline'
    COMMENT_CLOSE_DATE = 'commentCloseDate'
    STANDFIRST = 'standfirst'
    COMMENTABLE = 'commentable'
    IS_PREMODERATED = 'isPremoderated'
    LAST_MODIFIED = 'lastModified'
    NEWSPAPER_EDITION_DATE = 'newspaperEditionDate'
    LEGALLY_SENSITIVE = 'legallySensitive'
    ARTICLE = 'bodyText'

    # calculated fields
    HEADLINE_WORD_COUNT = 'headlineWordCount'
    STANDFIRST_WORD_COUNT = 'standfirstWordCount'
    ARTICLE_WORD_COUNT = 'articleWordCount'
    COMMENT_PERFORMANCE_WEEKLY = 'commentPerformanceWeekly'
    IN_TOP_TEN_PERCENT = 'inTopTenPercent'
    COMMENT_NUMBER = 'commentNumber'
    DAY_OF_WEEK = 'dayOfWeek'
    DAY_OF_YEAR = 'dayOfYear'
    HOUR = 'hour'
    MINUTE = 'minute'
    GENRE = 'genre'
