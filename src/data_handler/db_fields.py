from enum import Enum


class Authors(Enum):
    ID = 'author_id'
    COMMENT_AUTHOR = 'comment_author'


class Articles(Enum):
    ID = 'article_id'
    URL = 'article_url'


class Comments(Enum):
    ID = 'article_id'
    AUTHOR = 'author_id'
    COMMENT_ID = 'comment_id'
    COMMENT_TEXT = 'comment_text'
    PARENT_COMMENT_ID = 'parent_comment_id'
    TIMESTAMP = 'timestamp'
    UPVOTES = 'upvotes'


class GuardianApi(Enum):
    ID = 'article_id'
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
    TAGS = 'tags'


class GuardianApiError(Enum):
    ID = 'article_id'
    REASON = 'reason'


class LabelsView(Enum):
    ID = 'article_id'
    HEADLINE = 'headline'
    HEADLINE_WORD_COUNT = 'headline_word_count'
    ARTICLE = 'article'
    ARTICLE_WORD_COUNT = 'article_word_count'
    COMMENT_PERFORMANCE_WEEKLY = 'comment_performance_weekly'
    IN_TOP_TEN_PERCENT = 'in_top_ten_percent'
    COMMENT_COUNT = 'comment_count'
    UNIX_TIMESTAMP = 'unix_timestamp'
    DAY_OF_WEEK = 'day_of_week'
    DAY_OF_YEAR = 'day_of_year'
    HOUR = 'hour'
    MINUTE = 'minute'
