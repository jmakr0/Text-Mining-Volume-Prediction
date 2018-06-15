import keras


class CsvLogger(keras.callbacks.CSVLogger):

    def __init__(self, model, log_path):
        filename = '{}/{}'.format(log_path, 'training.csv')
        super().__init__(filename)
