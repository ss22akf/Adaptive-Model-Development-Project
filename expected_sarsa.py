import numpy as np
from agent_base import BaseAgent


class ExpectedSARSAAgent(BaseAgent):
    """
    Expected SARSA agent (on-policy RL algorithm).
    Updates Q-values using the expected value over all possible next actions:
    This combines:
        - SARSA (uses policy)
        - Q-learning (considers all actions)
    Result: smoother and more stable learning updates.
    """

    def update(self, state, action, reward, nextState):
        """
        Update Q-value using Expected SARSA rule.

        Parameters:
        - state : current state (s)
        - action : action taken (a)
        - reward : immediate reward (r)
        - nextState : next state (s')
        """

        # Q-values for all possible actions in next state
        qValues = self.qTable[nextState]

        # Identify best action (greedy action)
        bestAction = int(np.argmax(qValues))

        nA = len(qValues) # number of actions

        # Tuning factor for expectation
        effectiveEpsilon = self.epsilon * 0.5 
        expectedValue = 0.0

        # Compute expected value
        for a in range(nA):

            # Probability under epsilon-greedy policy
            prob = effectiveEpsilon / nA

            # Added to the best action 
            if a == bestAction:
                prob += (1 - effectiveEpsilon)

            # Weighted contribution of each action
            expectedValue += prob * qValues[a]

        # TD target using expected future value
        tdTarget = reward + self.gamma * expectedValue

        # Update Q-value
        self.qTable[state][action] += self.alpha * (
            tdTarget - self.qTable[state][action]
        )