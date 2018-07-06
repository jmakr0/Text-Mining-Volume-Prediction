import numpy as np


class CorrelationEvaluator:
    def __init__(self, model_1, model_2, data_1, data_2):
        self.model_1 = model_1
        self.model_2 = model_2

        self.data_1 = data_1
        self.data_2 = data_2

    def __call__(self):
        prediction_1 = self.model_1.predict(self.data_1)
        prediction_2 = self.model_2.predict(self.data_2)
        correlation = np.correlate(prediction_1[:,-1], prediction_2[:,-1])[0]
        print('correlation value: {}'.format(correlation))
