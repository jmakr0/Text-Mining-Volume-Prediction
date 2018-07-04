import os
from os.path import basename

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd


class CsvPlotter:
    X_AXIS_LABEL = "Epoch"

    def __init__(self, csv_filepath):
        self.csv_filepath = csv_filepath
        dir = os.path.dirname(os.path.abspath(csv_filepath))
        plot_name = '.'.join(basename(self.csv_filepath).split('.')[:-1]) + '_plot'
        self.plot_filepath = '{}/{}'.format(dir, plot_name)

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
        epochs = self.csv_data_frame['epoch'].values

        fig = plt.figure(figsize=scale)
        ax = fig.add_subplot(111)

        ax.grid(True, linestyle='-')
        ax.tick_params(labelcolor='black', labelsize='medium', width=3)

        plt.xlabel('epoch')

        # set interval of x axis
        loc = plticker.MultipleLocator(base=round(len(epochs)/10))
        ax.xaxis.set_major_locator(loc)

        for attribute, color, display_name in config:
            data_points = self.csv_data_frame[attribute].values
            ax.plot(epochs, data_points, color=color, linestyle='-', marker='o')

            if display_name:
                handle = mpatches.Patch(color=color, label=display_name)
            else:
                handle = mpatches.Patch(color=color, label=attribute)
            handles.append(handle)

        plt.legend(handles=handles)

        plt.savefig(self.plot_filepath)
