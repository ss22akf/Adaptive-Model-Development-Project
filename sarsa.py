from agent_base import BaseAgent

class SARSAAgent(BaseAgent):
    """
    SARSA agent (on-policy RL algorithm).
    Updates Q-values using the Bellman expectation equation
    Unlike Q-learning, SARSA uses the actual next action (a')selected by the current policy.
    """
    
    def update(self, state, action, reward, nextState, nextAction):
        """
        Update Q-value using SARSA rule.

        Parameters:
        - state : current state (s)
        - action : action taken (a)
        - reward : immediate reward (r)
        - nextState : next state (s')
        - nextAction : next action taken (a') under current policy
        """

        # Estimate of future value based on the action actually taken. This makes SARSA on-policy
        tdTarget = reward + self.gamma * self.qTable[nextState][nextAction]

        # Adjust Q(s, a) towards the observed target
        self.qTable[state][action] += self.alpha * (
            tdTarget - self.qTable[state][action]
        )