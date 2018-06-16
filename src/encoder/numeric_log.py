import math


class NumericLog:
    def __init__(self, max_value, base=1.5):
        self.max_value = max_value
        self.base = base

    def max_log_value(self):
        return math.floor(math.log(self.max_value, self.base))

    def __call__(self, number):
        return min(math.floor(math.log(number, self.base)), self.max_log_value())
