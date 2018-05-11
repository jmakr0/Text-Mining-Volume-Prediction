import yaml


class Settings:
    FILE = 'settings.yml'

    def __init__(self):
        with open(self.FILE, 'r') as yaml_file:
            self.config = yaml.load(yaml_file)

    def get_database_settings(self, database_name):
        return self.config.get('database')[database_name]

    def get_csv_file(self, name):
        return self.config.get('guardian-csv')[name]

    def get_guardian_api_keys(self):
        return self.config.get('guardian-api')['keys']
