import numpy as np


def calculate_class_weights(is_top_submission, output_names):
    classes, counts = np.unique(is_top_submission, return_counts=True)
    weights = len(is_top_submission) / counts
    class_weight_dict = dict(zip(classes, weights))

    return {name: class_weight_dict for name in output_names}
