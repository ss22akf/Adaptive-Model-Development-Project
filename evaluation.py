import numpy as np


def evaluateAgent(env, agent, episodes=100):
    """
    Evaluate a trained agent using a greedy policy.

    Parameters:
    - env : The warehouse simulation environment object
        
    - agent : Trained RL agent, Contains the learned Q-table
        
    - episodes : Number of evaluation runs
        
    Returns:
    - metrics : successRate, avgSteps, collisions, energy
        
    """

    # Initialise metrics
    metrics = {
        "successRate": 0.0,
        "avgSteps": 0.0,
        "collisions": 0.0,
        "energy": 0.0
    }


    # RUN evaluation episodes
    for _ in range(episodes):

        # Reset environment at the start of each episode
        state = env.reset()
        done = False

        # Run until episode terminates
        while not done:

            # Greedy action selection (no exploration)
            # Uses learned Q-values to choose best action
            action = int(np.argmax(agent.qTable[state]))

            # Execute action in environment
            state, reward, done = env.step(action)

        # COLLECT METRICS
        if env.robotPos == env.deliveryPos and env.hasPackage == 1:
            metrics["successRate"] += 1

        # Accumulate total steps taken in this episode
        metrics["avgSteps"] += env.steps

        # Accumulate number of collisions
        metrics["collisions"] += env.collisionCount

        # Accumulate steps as a proxy for energy consumption
        metrics["energy"] += env.steps

    # Divide accumulated values by number of episodes
    for k in metrics:
        metrics[k] /= episodes

    return metrics