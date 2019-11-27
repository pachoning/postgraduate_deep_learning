import numpy as np
import itertools

class DataDistribution:
    # TODO: 1. create `__init__` to initialize the class with random W and b
    def __init__(self, W=None, b=None):
        self.W = W or np.random.normal()
        self.b = b or np.random.normal()

    # TODO: 2. implement method `generate` to draw samples from this distribution
    def generate(self, n_samples = 1):
        x = np.random.normal(size = n_samples)
        y = self.W * x + self.b
        return x, y

    def __call__(self, n_samples = None):
        return self.generate(n_samples = n_samples)

fff = DataDistribution()
fff.W = 9
fff.b = 0
fff.generate(10)
