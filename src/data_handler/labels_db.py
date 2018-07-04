from src.data_handler.db_fields import LabelsView
from src.data_handler.postgres_db import PostgresDb


class LabelsDb(PostgresDb):

    def get_labeled_data(self):
        # don't use articles with exact 50 comments
        where_50_comments = 'NOT {}=50'.format(LabelsView.COMMENT_COUNT.value)

        # don't use articles published on 29th of February
        where_leapyear = "NOT (date_part('day'::text, to_timestamp({0}))=29 AND date_part('month'::text, to_timestamp({0}))=2)".format(
            LabelsView.UNIX_TIMESTAMP.value)

        where_param = ' AND '.join([where_50_comments, where_leapyear])

        # get data sorted by the article timestamp
        order_by_param = LabelsView.UNIX_TIMESTAMP.value + ' ASC'

        # INFO: using where1 and where2 combined reduces the number of tuples by ~ 2%
        return self.get_dicts(LabelsView, where_param=where_param, order_by_param=order_by_param)
