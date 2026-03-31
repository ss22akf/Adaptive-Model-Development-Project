import numpy as np
from agent_base import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent (off-policy RL algorithm).
    Learns the optimal Q-values using the Bellman optimality equation:
    """

    def update(self, state, action, reward, nextState):
        """
        Update Q-value using Q-learning rule.

        Parameters:
        - state : current state (s)
        - action : action taken (a)
        - reward : immediate reward (r)
        - nextState : next state (s')
        """

        # Estimate of optimal future value, makes Q-learning off-policy (does not depend on next action taken)
        bestNext = np.max(self.qTable[nextState])

        # Temporal Difference (TD) target:
        tdTarget = reward + self.gamma * bestNext
        
        # Adjust current Q-value towards the target
        self.qTable[state][action] += self.alpha * (
            tdTarget - self.qTable[state][action]
        )