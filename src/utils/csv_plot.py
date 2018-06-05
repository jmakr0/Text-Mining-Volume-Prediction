import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd


class Csv_plotter():
    x_axis_label = "Epoch"

    def __init__(self, csv_filepath):
        self.csv_data_frame = pd.read_csv(csv_filepath)

    def plot_csv_data(self, config, filename="plot", scale=(8, 4)):
        """
        plots subset of data from csv file
        :param config: config should be a list of entries with the following structure: (<attribute_to_plot>,<color_for_plot>,<attribute_display_name>)
        :param filename: the filename for the plot
        :param scale: the scale of the plot as a tupel (<width_scale>,<height_scale>)
        """
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

        plt.savefig(filename)

    def _get_column_as_numpy_array(self, column_name):
        return self.csv_data_frame[column_name].values
