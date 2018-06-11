import os

from keras.callbacks import CSVLogger


class CustomCsvLogger(CSVLogger):

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        super().on_train_begin()

