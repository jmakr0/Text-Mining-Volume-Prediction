import os
from time import strftime, gmtime

from src.utils.logging.callbacks.config_logger import ConfigLogger
from src.utils.logging.callbacks.csv_logger import CsvLogger
from src.utils.logging.callbacks.csv_plotter import CsvPlotter
from src.utils.settings import Settings


class CallbackBuilder():

    def __init__(self, model, default_parameters, hyper_parameters, callback_classes):
        super().__init__()

        self.model = model
        self.default_parameters = default_parameters
        self.hyper_parameters = hyper_parameters
        self.callback_classes = callback_classes

        self.active_callbacks = []

    def __call__(self):
        timestamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        settings = Settings()
        log_path = '{}/{}_{}'.format(settings.get_training_root_dir(), self.model.name, timestamp)
        os.makedirs(log_path, exist_ok=True)

        if CsvLogger in self.callback_classes:
            csv_logger = CsvLogger(self.model.name, log_path)
            self.active_callbacks.append(csv_logger)

        if CsvPlotter in self.callback_classes:
            assert CsvLogger in self.callback_classes

            plotter = CsvPlotter(self.model, log_path)
            self.active_callbacks.append(plotter)

        if ConfigLogger in self.callback_classes:
            config_logger = ConfigLogger(self.model, self.default_parameters, self.hyper_parameters, log_path)
            self.active_callbacks.append(config_logger)

        return self.active_callbacks
