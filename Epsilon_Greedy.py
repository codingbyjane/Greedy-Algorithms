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
        if np.random.random() > self.epsilon: # np.random.random() generates a random float between 0 and 1. If this random number is greater than epsilon, we choose to exploit (select the best known action). Otherwise, we explore (select a random action).
            return self.ind_max(self.values) # Exploit: choose the action with the highest estimated value. Passing the values list as an argument to the ind_max function to find the index of the action with the highest estimated value. This is the action that will be selected for exploitation.
        else:
            return np.random.randrange(len(self.values)) # Explore: choose a random action
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1 # Increment the count of how many times the chosen arm has been selected. This is important for updating the estimated value of that arm.
        n = self.counts[chosen_arm] # Get the updated count for the chosen arm
        value = self.values[chosen_arm] # Get the current estimated value for the chosen arm

        new_value = ((n - 1) / n) * value + (1 / n) * reward # Update the estimated value of the chosen arm using an incremental formula. This formula calculates a new average based on the previous average (value) and the new reward received from selecting that arm.
        self.values[chosen_arm] = new_value # Update the estimated value for the chosen arm in the values list with the newly calculated value. This allows the algorithm to improve its estimates of the performance of each arm over time as it receives more rewards from selecting different arms.
