from enum import Enum


class Authors(Enum):
    AUTHOR_ID = 'author_id'
    COMMENT_AUTHOR = 'comment_author'


class Articles(Enum):
    ARTICLE_ID = 'article_id'
    URL = 'article_url'


class Comments(Enum):
    ARTICLE_ID = 'article_id'
    AUTHOR = 'author_id'
    COMMENT_ID = 'comment_id'
    COMMENT_TEXT = 'comment_text'
    PARENT_COMMENT_ID = 'parent_comment_id'
    TIMESTAMP = 'timestamp'
    UPVOTES = 'upvotes'