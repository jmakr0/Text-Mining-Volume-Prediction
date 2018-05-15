import psycopg2
from psycopg2._psycopg import AsIs


class PostgresDb:
    def __init__(self, connection_dict):
        settings = " ".join(['%s=%s' % (key, value) for (key, value) in connection_dict.items()])
        self.conn = psycopg2.connect(settings)
        self.cur = self.conn.cursor()

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
            print(e)
