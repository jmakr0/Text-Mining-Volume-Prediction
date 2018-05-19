import numpy as np
import os
from random import random, sample, seed

from src.data_handler.db_fields import LabelsView
from src.data_handler.labels_db import LabelsDb

# Todo: Embedding into settings

embeddings_path = '/Volumes/Extreme 510/Data/glove.6B.50d.txt'

# Todo: Encapsulate data into object

titles = []
hours = []
minutes = []
dayofweeks = []
dayofyears = []
is_top_submission = []

labels_db = LabelsDb()

labeled_articles = labels_db.get_labeled_data()

# data

# ToDo: Headline and other datat needs to be in the view on the db server
for article in labeled_articles:
    titles.append(article[LabelsView.HEADLINE.value])
    hours.append(article[LabelsView.HOUR.value])
    minutes.append(article[LabelsView.MINUTE.value])
    dayofweeks.append(article[LabelsView.DAY_OF_WEEK.value])
    dayofyears.append(article[LabelsView.DAY_OF_YEAR.value])
    is_top_submission.append(article[LabelsView.IN_TOP_TEN_PERCENT.value])

# transform data

titles = np.array(titles)
hours = np.array(hours, dtype=int)
minutes = np.array(minutes, dtype=int)
dayofweeks = np.array(dayofweeks, dtype=int)
dayofyears = np.array(dayofyears, dtype=int)
is_top_submission = np.array(is_top_submission, dtype=int)

# Test print

print(titles[0:2])
print(titles.shape)
print(hours[0:2])
print(minutes[0:2])
print(dayofweeks[0:2])
print(dayofyears[0:2])
print(is_top_submission[0:2])

no_info_rate = 1 - np.mean(is_top_submission)