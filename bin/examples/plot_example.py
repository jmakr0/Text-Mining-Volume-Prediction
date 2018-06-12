import os

from src.utils.csv_plot import CsvPlotter

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.abspath(__file__))
    plotter = CsvPlotter(script_path + '/../../src/training.csv')
    config = [('main_output_f1', (0.1, 0.0, 0.9), 'f1-score'), ('val_main_output_f1', 'g', 'validation f1-score')]
    plotter.plot_csv_data(config, filename='test_plot', scale=(8, 4))