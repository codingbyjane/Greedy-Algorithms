# Importing Dependencies

import numpy as np
import matplotlib.pyplot as plt

# Epsilon Greedy Algorithm
class EpsilonGreedy:
    def __init__(self, epsilon, count, values): # The constructor method initializes the EpsilonGreedy class with three parameters: epsilon, count, and values. These parameters are essential for the functioning of the algorithm.
        self.epsilon = epsilon # Exploration rate, a.k.a the frequency with which one of the options will be chosen (explore/exploit). A number between 0 and 1, where 0 means always exploit and 1 means always explore.
        self.counts = count # Number of times each action has been taken / option has been chosen
        self.values = values # Estimated performance value of each action / option (estimated reward value of a particular arm / action). This is a list that holds the estimated value for each action/option based on the rewards received from selecting those actions in the past. The algorithm uses these values to make informed decisions about which action to select during exploitation.

    def ind_max(self, reward_list): # Function to find the index of the maximum value in a list, a.k.a, the index of the best performing arm / option
        max_value = max(reward_list)
        return reward_list.index(max_value) # Return the index of the maximum value in the list. This is used to select the action with the highest estimated reward value during exploitation. So this method returns the index of the best-performing item in the reward_list.

    def select_arm(self):
        if np.random.random() > self.epsilon: # np.random.random() generates a random float between 0 and 1. If this random number is greater than epsilon, we choose to exploit (select the best known action). Otherwise, we explore (select a random action).
            return self.ind_max(self.values) # Exploit: choose the action with the highest estimated value. Passing the values list as an argument to the ind_max function to find the index of the action with the highest estimated value. This is the action that will be selected for exploitation.
        else:
            return np.random.randrange(len(self.values)) # Explore: choose a random action
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1 # Increment the count of how many times the chosen arm has been selected. This is important for updating the estimated value of that arm.
        n = self.counts[chosen_arm] # Get the updated count for the chosen arm. Stores the updated count in a local variable.
        value = self.values[chosen_arm] # Get the current estimated reward value for the chosen arm. This is the old estimate before incorporating the new reward.

        new_value = ((n - 1) / n) * value + (1 / n) * reward # Update the estimated value of the chosen arm using an incremental formula. This formula calculates a new average based on the previous average (value) and the new reward received from selecting that arm.
        self.values[chosen_arm] = new_value # Update the estimated value for the chosen arm in the values list with the newly calculated value. This allows the algorithm to improve its estimates of the performance of each arm over time as it receives more rewards from selecting different arms.
        return new_value # Return the updated estimated value for the chosen arm after incorporating the new reward. This can be useful for tracking the performance of each arm over time.
    
# Example of creating an instance of the EpsilonGreedy class with an exploration rate of 0.1 and three arms (options) initialized with zero counts and zero estimated values.
epsilon_greedy_agent = EpsilonGreedy(epsilon=0.1, count=[0, 0, 0], values=[0.0, 0.0, 0.0]) # Initializing the EpsilonGreedy agent with an exploration rate of 0.1 (10%), and three arms (options) that have been selected 0 times and have an estimated value of 0.0. This sets up the agent to start making decisions based on the epsilon-greedy strategy.

arm = epsilon_greedy_agent.select_arm() # Select an arm (option) using the select_arm method of the EpsilonGreedy agent. This will return the index of the selected arm based on the epsilon-greedy strategy, which may involve either exploitation (choosing the best known arm) or exploration (choosing a random arm).
print(f"Selected arm: {arm}") # Print the index of the selected arm to the console. This allows us to see which arm was chosen by the epsilon-greedy algorithm based on the current estimates and exploration rate.

reward = 1.0 # Simulate receiving a reward of 1.0 for selecting the chosen arm. In a real application, this reward would come from the environment based on the action taken by the agent.
updated_value = epsilon_greedy_agent.update(chosen_arm=arm, reward=reward) # Update the estimated value of the chosen arm using the update method of the EpsilonGreedy agent. This method takes the index of the chosen arm and the reward received as arguments, and it updates the estimated value for that arm based on the new reward.