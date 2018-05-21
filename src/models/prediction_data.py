from src.data_handler.db_fields import LabelsView
from src.data_handler.labels_db import LabelsDb

import numpy as np

class PredictionData:

    def __init__(self):
        self.db = LabelsDb()
        self.titles = []
        self.hours = []
        self.minutes = []
        self.dayofweeks = []
        self.dayofyears = []
        self.is_top_submission = []
        self.no_info_rate = 0

    def _load_data(self):
        labeled_articles = self.db.get_labeled_data()

        for article in labeled_articles:
            self.titles.append(article[LabelsView.HEADLINE.value])
            self.hours.append(article[LabelsView.HOUR.value])
            self.minutes.append(article[LabelsView.MINUTE.value])
            self.dayofweeks.append(article[LabelsView.DAY_OF_WEEK.value])
            self.dayofyears.append(article[LabelsView.DAY_OF_YEAR.value])
            self.is_top_submission.append(1 if article[LabelsView.IN_TOP_TEN_PERCENT.value] == 'TRUE' else 0)

    def _transform_to_arrays(self):
        self.titles = np.array(self.titles)
        self.hours = np.array(self.hours, dtype=int)
        self.minutes = np.array(self.minutes, dtype=int)
        self.dayofweeks = np.array(self.dayofweeks, dtype=int)
        self.dayofyears = np.array(self.dayofyears, dtype=int)
        self.is_top_submission = np.array(self.is_top_submission, dtype=int)
        # dayofyears is 1-indexed, so must subtract 1.
        self.dayofyears = self.dayofyears -1

    def prepare(self):
        self._load_data()
        self._transform_to_arrays()
        self.no_info_rate = 1 - np.mean(self.is_top_submission)

    def test_print(self):
        print(self.titles[0:2])
        print(self.titles.shape)
        print(self.hours[0:2])
        print(self.minutes[0:2])
        print(self.dayofweeks[0:2])
        print(self.dayofyears[0:2])
        print(self.is_top_submission[0:2])
        print(self.no_info_rate)

    def get_headlines(self):
        return self.titles

    def get_hours(self):
        return self.hours

    def get_minutes(self):
        return self.minutes

    def get_dayofweeks(self):
        return self.dayofweeks

    def get_dayofyears(self):
        return self.dayofyears

    def get_is_top_submission(self):
        return self.is_top_submission