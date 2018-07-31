from keras.callbacks import Callback


class ModelSaver(Callback):

    def __init__(self, model, log_filepath):
        super().__init__()

        self.model = model
        self.model_filepath = '{}/{}'.format(log_filepath, 'model.h5')

    def on_train_end(self, logs=None):
        self.model.save(self.model_filepath)
