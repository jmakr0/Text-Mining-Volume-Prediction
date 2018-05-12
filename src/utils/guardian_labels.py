from enum import Enum


class GuardianLabels(Enum):
    HEADLINE = 'headline'
    COMMENT_CLOSE_DATE = 'commentCloseDate'
    STANDFIRST = 'standfirst'
    COMMENTABLE = 'commentable'
    IS_PREMODERATED = 'isPremoderated'
    LAST_MODIFIED = 'lastModified'
    NEWSPAPER_EDITION_DATE = 'newspaperEditionDate'
    LEGALLY_SENSITIVE = 'legallySensitive'
    ARTICLE = 'bodyText'
