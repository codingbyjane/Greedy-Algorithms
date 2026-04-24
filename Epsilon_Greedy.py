# Importing Dependencies

import numpy as np
import matplotlib.pyplot as plt

# Epsilon Greedy Algorithm
class EpsilonGreedy:
    def __init__(self, epsilon, count, values):
        self.epsilon = epsilon # Exploration rate, a.k.a the frequency with which one of the options will be chosen (explore/exploit). A number between 0 and 1, where 0 means always exploit and 1 means always explore.
        self.count = count # Number of times each action has been taken / option has been chosen
        self.values = values # Estimated performance value of each action / option (estimated reward value)

    def ind_max(self, x): # Function to find the index of the maximum value in a list, a.k.a, the index of the best performing arm / option
        max_value = max(x)
        return x.index(max_value) # Return the index of the maximum value in the list. This is used to select the action with the highest estimated value during exploitation.

    def select_arm(self):
        if np.random.random() > self.epsilon:
            return self.ind_max(self.values) # Exploit: choose the action with the highest estimated value
        else:
            return np.random.randrange(len(self.values)) # Explore: choose a random action
