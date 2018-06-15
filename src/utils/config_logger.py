import os

from keras.callbacks import Callback

from src.utils.settings import Settings


class ConfigLogger():

    def __init__(self, model, timestamp):
        settings = Settings()

        self.filepath = settings.get_training_root_dir() + model.name + "_" + timestamp

        filename = self.filepath + "/config.txt"

        self.log_filename = filename

        model.summary(print_fn=self._handle_summary_print)
        self.model = model

    def log_training_config(self, model_description, hyperparameters):
        log_text = ""
        log_text += "\n" + "MODEL NAME:\n" + self.model.name + "\n"
        log_text += "\n" + "MODEL DESCRIPTION:\n" + model_description + "\n"
        log_text += "HYPERPARAMETERS:\n" + str(hyperparameters)

        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        with open(self.log_filename, "w") as f:
            f.write(log_text)

    def _handle_summary_print(self, s):
        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        with open(self.filepath + '/model_summary.txt', 'a') as f:
            print(s, file=f)


class ConfigLoggerCallback(Callback):

    def __init__(self, model, timestamp, model_description, hyperparameters):
        super(ConfigLoggerCallback, self).__init__()
        self.config_logger = ConfigLogger(model, timestamp)
        self.model_description = model_description
        self.hyperparameters = hyperparameters


    def on_train_end(self, logs=None):
        self.config_logger.log_training_config(self.model_description, self.hyperparameters)
