from src.data_handler.db_fields import LabelsView
from src.data_handler.postgres_db import PostgresDb


class LabelsDb(PostgresDb):

    def get_labeled_data(self):
        return self.get_dicts(LabelsView, LabelsView.UNIX_TIMESTAMP.value + ' ASC')
