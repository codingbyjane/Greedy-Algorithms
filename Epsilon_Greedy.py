# Importing Dependencies

import numpy as np
import matplotlib.pyplot as plt

# Epsilon Greedy Algorithm
class EpsilonGreedy:
    def __init__(self, epsilon, count, values):
        self.epsilon = epsilon # Exploration rate, a.k.a the frequency with which one of the options will be chosen (explore/exploit)
        self.count = count # Number of times each action has been taken / option has been chosen
        self.values = values # Estimated performance value of each action / option

    def ind_max(self, x): # Function to find the index of the maximum value in a list
        m = max(x)
        return x.index(m) # Return the index of the maximum value in the list

    def select_arm(self):
        if np.random.random() > self.epsilon:
            return self.ind_max(self.values) # Exploit: choose the action with the highest estimated value
        else:
            return np.random.randrange(len(self.values)) # Explore: choose a random action
