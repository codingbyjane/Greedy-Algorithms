# Importing Dependencies

import numpy as np
import matplotlib.pyplot as plt

# Epsilon Greedy Algorithm
class EpsilonGreedy:
    def __init__(self, epsilon, count, values):
        self.epsilon = epsilon
        self.count = count
        self.values = values
    return