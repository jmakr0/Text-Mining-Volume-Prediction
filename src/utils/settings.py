import yaml


class Settings:
    FILE = 'settings.yml'

    def __init__(self):
        with open(self.FILE, 'r') as yaml_file:
            self.config = yaml.load(yaml_file)

    def get_database_settings(self):
        return self.config.get('database')

    def get_glove_embedding(self):
        return self.config.get('embeddings')['glove']

    def get_doc2vec_dir(self):
        return self.config.get('models')['doc2vec_dir']

    def get_csv_file(self, name):
        return self.config.get('guardian_csv')[name]

    def get_guardian_api_keys(self):
        return self.config.get('guardian_api')['keys']

    def get_training_root_dir(self):
        return self.config.get('training_results')['root_dir']

    def get_training_parameters(self, name=None):
        if name == None:
            return self.config.get('training_parameters')
        return self.config.get('training_parameters')[name]

    def get_network_parameters(self, name=None):
        if name == None:
            return self.config.get('network_parameters')
        return self.config.get('network_parameters')[name]
