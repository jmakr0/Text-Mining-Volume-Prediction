import os

from keras.callbacks import CSVLogger

from src.utils.settings import Settings


class CustomCsvLogger(CSVLogger):

    def __init__(self, modelname, timestamp):
        settings = Settings()

        filepath = settings.get_training_root_dir() + modelname + "_" + timestamp + "/"

        filename = filepath + "training.csv"

        self.csv_file = filename

        super(CustomCsvLogger, self).__init__(filename)

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        super().on_train_begin()
