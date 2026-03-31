import numpy as np
from collections import defaultdict


class BaseAgent:
    """
    Base class for reinforcement learning agents.

    This class implements:
    - Q-table storage
    - Epsilon-greedy action selection strategy

    """

    def __init__(self, actionSize, alpha, gamma, epsilon, rng):

        # Q-table
        # Maps each state to an array of action Q-values
        # Default initial values = 0 for all actions
        self.qTable = defaultdict(lambda: np.zeros(actionSize, dtype=float))

        self.alpha = alpha              # Learning rate (controls how quickly Q-values are updated)
        self.gamma = gamma              # Discount factor (importance of future rewards)
        self.epsilon = epsilon          # Exploration rate for epsilon-greedy policy
        self.rng = rng                  # numpy random generator, ensures reproducibility across runs
        self.actionSize = actionSize    # Number of possible actions in the environment

    def selectAction(self, state):
        """
        Select an action using epsilon-greedy policy.

        With probability epsilon:
            - choose a random action (exploration)

        With probability (1 - epsilon):
            - choose the best action based on current Q-values (exploitation)

        Parameters:
        - state : 
            Current state of the environment

        Returns:
        - action :
            Selected action index
        """
 
        # Exploration: random action
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.actionSize))

        # Exploitation: best known action
        return int(np.argmax(self.qTable[state]))