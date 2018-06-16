import psycopg2
import psycopg2.extras
from psycopg2._psycopg import AsIs

from src.utils.settings import Settings


class PostgresDb:
    def __init__(self):
        settings = Settings()
        connection_dict = settings.get_database_settings()
        settings = " ".join(['%s=%s' % (key, value) for (key, value) in connection_dict.items()])
        self.conn = psycopg2.connect(settings)
        self.cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    def get_dict(self, db_fields_enum, id):
        try:
            table = db_fields_enum.__name__.lower()

            select = 'SELECT * FROM {0} WHERE {1}={2}'.format(table, db_fields_enum.ID.value, id)
            self.cur.execute(select)
            return self.cur.fetchone()
        except Exception as e:
            print("cant get dict")
            print(e)

    def get_dicts(self, db_fields_enum):
        try:
            table = db_fields_enum.__name__.lower()
            select = 'SELECT * FROM {} ORDER BY {} ASC'.format(table, 'unix_timestamp')
            self.cur.execute(select)
            return self.cur.fetchall()
        except Exception as e:
            print("cant get dicts")
            print(e)

    def insert_dict(self, db_fields_enum, dict):
        try:
            table = db_fields_enum.__name__.lower()

            none_empty_columns = list(filter(lambda e: dict[e.value], db_fields_enum))
            columns = list(map(lambda e: e.value, none_empty_columns))
            values = [dict[col] for col in columns]

            insert_statement = 'insert into {} (%s) values %s'.format(table)
            self.cur.execute(insert_statement, (AsIs(','.join(columns)), tuple(values)))
            self.conn.commit()
        except Exception as e:
            print("cant insert dict")
            print(e)
