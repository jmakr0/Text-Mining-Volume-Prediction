from keras.callbacks import Callback


class ConfigLogger(Callback):

    def __init__(self, model, hyper_parameters, log_path):
        super().__init__()

        self.hyper_parameters = hyper_parameters
        self.log_path = log_path
        self.filename = '{}/{}'.format(log_path, 'config.txt')

        model.summary(print_fn=self._handle_summary_print)
        self.model = model

    def _handle_summary_print(self, s):
        with open('{}/{}'.format(self.log_path, 'model_summary.txt'), 'a') as f:
            print(s, file=f)

    def on_train_end(self, logs=None):
        log_text = 'MODEL NAME:\n{}\nHYPER PARAMETERS:\n{}'.format(self.model.name, str(self.hyper_parameters))
        with open(self.filename, "w") as f:
            f.write(log_text)
