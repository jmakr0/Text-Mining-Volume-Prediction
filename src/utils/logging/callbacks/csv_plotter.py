from keras.callbacks import Callback

from src.utils.csv_plot import CsvPlotter as Plotter


class CsvPlotter(Callback):

    def __init__(self, model, log_filepath):
        super().__init__()

        last_layer = model.output_layers[-1].name
        last_layer_f1 = 'f1' if len(model.output_layers) == 1 else last_layer + '_f1'
        last_layer_val_f1 = 'val_' + last_layer_f1

        self.config = [(last_layer_f1, (0.1, 0.0, 0.9), 'f1-score'),
                       (last_layer_val_f1, 'g', 'validation f1-score')]
        csv_filepath = '{}/{}'.format(log_filepath, 'training.csv')
        self.csv_plotter = Plotter(csv_filepath)

    def on_train_end(self, logs=None):
        self.csv_plotter.plot_csv_data(self.config)
