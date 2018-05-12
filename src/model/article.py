import dateutil.parser

from src.utils.labels import Labels
from src.utils.utils import word_count


class Article:
    def __init__(self, article_id, article_dict):
        self.article_id = article_id
        self.article_dict = {}
        for label in list(Labels):
            if label.value in article_dict:
                self.article_dict[label.value] = article_dict[label.value]
            else:
                self.article_dict[label.value] = None

        self._create_labels()

    def _create_labels(self):
        for label in list(Labels):
            if self.article_dict[label.value] is None:
                method_name = '_create_label_' + label.value
                if method_name in dir(self):
                    getattr(self, method_name)()

    def _create_label_headline_word_count(self):
        self.article_dict[Labels.HEADLINE_WORD_COUNT.value] = word_count(self.article_dict[Labels.HEADLINE.value])

    def _create_label_standfirst_word_count(self):
        self.article_dict[Labels.STANDFIRST_WORD_COUNT.value] = word_count(self.article_dict[Labels.STANDFIRST.value])

    def _create_label_article_word_count(self):
        self.article_dict[Labels.ARTICLE_WORD_COUNT.value] = word_count(self.article_dict[Labels.ARTICLE.value])

    def _create_label_unix_timestamp(self):
        date = dateutil.parser.parse(self.article_dict[Labels.LAST_MODIFIED.value])
        self.article_dict[Labels.UNIX_TIMESTAMP.value] = int(date.timestamp())


    def _create_label_day_of_week(self):
        date = dateutil.parser.parse(self.article_dict[Labels.LAST_MODIFIED.value])
        self.article_dict[Labels.DAY_OF_WEEK.value] = date.weekday()

    def _create_label_day_of_year(self):
        # TODO
        pass

    def _create_label_hour(self):
        date = dateutil.parser.parse(self.article_dict[Labels.LAST_MODIFIED.value])
        self.article_dict[Labels.HOUR.value] = date.hour

    def _create_label_minute(self):
        date = dateutil.parser.parse(self.article_dict[Labels.LAST_MODIFIED.value])
        self.article_dict[Labels.MINUTE.value] = date.minute

    def _create_label_genre(self):
        # TODO
        pass

    def __getitem__(self, item):
        if item in self.article_dict:
            return self.article_dict[item]
