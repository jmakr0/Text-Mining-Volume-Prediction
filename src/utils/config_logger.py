import os

from keras.callbacks import Callback

from src.utils.settings import Settings


class ConfigLogger():

    def __init__(self, modelname, timestamp):
        settings = Settings()

        filepath = settings.get_training_root_dir() + modelname + "_" + timestamp

        filename = filepath + "/config.txt"

        self.log_filename = filename

    def log_training_config(self, model_name, model_description, hyperparameters):
        log_text = ""

        log_text += "MODEL NAME:\n" + model_name + "\n"
        log_text += "\n" + "MODEL DESCRIPTION:\n" + model_description + "\n"
        log_text += "HYPERPARAMETERS:\n" + str(hyperparameters)

        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        with open(self.log_filename, "w") as f:
            f.write(log_text)

class ConfigLoggerCallback(Callback):

    def __init__(self, modelname, timestamp, model_description, hyperparameters):
        super(ConfigLoggerCallback, self).__init__()
        self.config_logger = ConfigLogger(modelname, timestamp)
        self.model_name = modelname
        self.model_description = model_description
        self.hyperparameters = hyperparameters


    def on_train_end(self, logs=None):
        self.config_logger.log_training_config(self.model_name, self.model_description, self.hyperparameters)
