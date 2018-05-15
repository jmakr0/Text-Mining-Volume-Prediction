from src.data_handler.db_fields import Labels, GuardianApi
from src.utils.utils import word_count


class Article:
    def __init__(self, download, label_db):
        self.download = download
        self.label_db = label_db
        self.labels = {}
        self._create_labels()

    def get_labels(self):
        return self.labels

    def _create_labels(self):
        for label in list(Labels):
            method_name = '_create_label_' + label.value
            if method_name in dir(self):
                getattr(self, method_name)()

    def _create_label_article_id(self):
        self.labels[Labels.ID.value] = self.download[GuardianApi.ID.value]

    def _create_label_headline_word_count(self):
        self.labels[Labels.HEADLINE_WORD_COUNT.value] = word_count(self.download[GuardianApi.HEADLINE.value.lower()])

    def _create_label_standfirst_word_count(self):
        self.labels[Labels.STANDFIRST_WORD_COUNT.value] = word_count(self.download[GuardianApi.STANDFIRST.value.lower()])

    def _create_label_article_word_count(self):
        self.labels[Labels.ARTICLE_WORD_COUNT.value] = word_count(self.download[GuardianApi.ARTICLE.value.lower()])

    def _create_label_unix_timestamp(self):
        date = self.download[GuardianApi.PUBLICATION_DATE.value.lower()]
        self.labels[Labels.UNIX_TIMESTAMP.value] = int(date.timestamp())

    def _create_label_day_of_week(self):
        date = self.download[GuardianApi.PUBLICATION_DATE.value.lower()]
        self.labels[Labels.DAY_OF_WEEK.value] = date.weekday()

    def _create_label_day_of_year(self):
        date = self.download[GuardianApi.PUBLICATION_DATE.value.lower()]
        self.article_dict[Labels.DAY_OF_WEEK.value] = date.timetuple().tm_yday

    def _create_label_hour(self):
        date = self.download[GuardianApi.PUBLICATION_DATE.value.lower()]
        self.labels[Labels.HOUR.value] = date.hour

    def _create_label_minute(self):
        date = self.download[GuardianApi.PUBLICATION_DATE.value.lower()]
        self.labels[Labels.MINUTE.value] = date.minute

    def _create_label_minute(self):
        date = self.download[GuardianApi.PUBLICATION_DATE.value.lower()]
        self.labels[Labels.MINUTE.value] = date.minute

    def _create_label_comment_count(self):
        self.labels[Labels.COMMENT_COUNT.value] = self.label_db.get_comment_count(self.download[GuardianApi.ID.value])

    def _create_label_in_top_ten_percent(self):
        # Todo: SQL-Magic
        pass

    def _create_label_comment_performance_weekly(self):
        # Todo: SQL-Magic
        pass

    def __getitem__(self, item):
        if item in self.labels:
            return self.labels[item]
