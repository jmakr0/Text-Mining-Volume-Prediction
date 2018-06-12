import os
from os.path import basename

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd
from keras.callbacks import Callback


class CsvPlotter():
    X_AXIS_LABEL = "Epoch"

    def __init__(self, csv_filepath):
        self.csv_filepath = csv_filepath
        self.default_plot_filename = self._create_default_plot_filename()


    def plot_csv_data(self, config, filename="", scale=(8, 4)):
        """
        plots subset of data from csv file
        :param config: config should be a list of entries with the following structure: (<attribute_to_plot>,<color_for_plot>,<attribute_display_name>)
        :param filename: the filename for the plot
        :param scale: the scale of the plot as a tupel (<width_scale>,<height_scale>)
        """
        self.csv_data_frame = pd.read_csv(self.csv_filepath)

        handles = []

        # get epochs as data for x axis
        epochs = self._get_column_as_numpy_array("epoch")

        fig = plt.figure(figsize=scale)
        ax = fig.add_subplot(111)

        ax.grid(True, linestyle='-')
        ax.tick_params(labelcolor='black', labelsize='medium', width=3)

        plt.xlabel('epoch')

        # set interval of x axis
        loc = plticker.MultipleLocator(base=2.0)
        ax.xaxis.set_major_locator(loc)

        for attribute, color, display_name in config:
            data_points = self._get_column_as_numpy_array(attribute)
            ax.plot(epochs, data_points, color=color, linestyle='-', marker='o')

            if display_name:
                handle = mpatches.Patch(color=color, label=display_name)
            else:
                handle = mpatches.Patch(color=color, label=attribute)
            handles.append(handle)

        plt.legend(handles=handles)

        filepath = self._get_filepath_to_save(filename)

        plt.savefig(filepath)

    def _get_column_as_numpy_array(self, column_name):
        return self.csv_data_frame[column_name].values

    def _get_filepath_to_save(self, filename):
        if filename == "":
            filename = self.default_plot_filename

        filepath = os.path.dirname(os.path.abspath(self.csv_filepath))

        return filepath + "/" + filename


    def _create_default_plot_filename(self):
        csv_filename = basename(self.csv_filepath)
        # remove the file ending
        without_ending = csv_filename.split('.')[0]
        return str(without_ending) + '_plot'


class CSVPlotterCallback(Callback):

    def __init__(self, csv_filepath, config):
        super(CSVPlotterCallback, self).__init__()
        self.csv_plotter = CsvPlotter(csv_filepath)
        self.config = config


    def on_train_end(self, logs=None):
        self.csv_plotter.plot_csv_data(self.config)



