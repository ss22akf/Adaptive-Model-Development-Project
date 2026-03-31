import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

from warehouse_env import WarehouseEnv
from q_learning import QLearningAgent
from sarsa import SARSAAgent
from expected_sarsa import ExpectedSARSAAgent

from evaluation import evaluateAgent
from plotting import plotRewards
from comparison import compareResults
from matplotlib.patches import Patch

# Central configuration for training hyperparameters.
DEFAULT_CONFIG = {
    "alpha": 0.1,               # learning rate
    "gamma": 0.99,              # discount factor
    "epsilon": 1.0,             # initial exploration rate
    "epsilonDecay": 0.997,      # epsilon decay after each episode
    "minEpsilon": 0.05,         # minimum exploration threshold
    "episodes": 10000,          # default number of training episodes
}

MODEL_MAP = {
    "q_learning": "q",
    "sarsa": "sarsa",
    "expected_sarsa": "expected"
}

# Agent creation
def createAgent(modelType, actionSize, config, rng):
    """
    Create and return the requested RL agent.
   
    Parameters:
    - modelType : 
        Algorithm type ('q', 'sarsa', 'expected')

    - actionSize :
        Number of available actions in the environment

    - config : 
        Training hyperparameters

    - rng : 
        numpy random generator, used for reproducibility
    """

    # Create Q-Learning agent
    if modelType == "q":
        return QLearningAgent(
            actionSize,
            config["alpha"],
            config["gamma"],
            config["epsilon"],
            rng
        )

    # Create SARSA agent
    elif modelType == "sarsa":
        return SARSAAgent(
            actionSize,
            config["alpha"],
            config["gamma"],
            config["epsilon"],
            rng
        )

    # Create Expected SARSA agent
    elif modelType == "expected":
        return ExpectedSARSAAgent(
            actionSize,
            config["alpha"],
            config["gamma"],
            config["epsilon"],
            rng
        )

    # Reject invalid model types
    else:
        raise ValueError("Invalid model type")



# Train a single model per seed 
def trainOnce(modelType, seed, config):
    """
    Train one model for one random seed.

    Parameters:
    - modelType :
        Selected RL algorithm
    - seed :
        Random seed for reproducibility
    - config : 
        Training settings

    Returns:
    - agent : trained agent
    - rewards : Total reward per episode
        
    - metrics : Evaluation metrics after training
    """

    # Create random generator and environment with same seed
    rng = np.random.default_rng(seed)
    env = WarehouseEnv(seed=seed)

    # Create the selected agent
    agent = createAgent(modelType, env.actionSpace, config, rng)

    # Store total reward from each episode
    rewards = []


    # Training loop
    for ep in range(config["episodes"]):
        state = env.reset()
        done = False
        totalReward = 0.0

        # Select first action
        action = agent.selectAction(state)

        while not done:
            # Apply action to environment
            nextState, reward, done = env.step(action)
            totalReward += reward

            # SARSA update uses the next chosen action
            if modelType == "sarsa":
                nextAction = agent.selectAction(nextState)
                agent.update(state, action, reward, nextState, nextAction)
                action = nextAction

            # Q-learning and Expected SARSA compute update directly from next state
            else:
                agent.update(state, action, reward, nextState)
                action = agent.selectAction(nextState)

            # Move to next state
            state = nextState

        # Store episode reward for learning curve
        rewards.append(totalReward)

        # Decay exploration rate after each episode
        agent.epsilon = max(
            config["minEpsilon"],
            agent.epsilon * config["epsilonDecay"]
        )

    # Evaluate trained agent
    evalEnv = WarehouseEnv(seed=seed)
    metrics = evaluateAgent(evalEnv, agent, episodes=100)

    return agent, rewards, metrics



# Aggregate metrics across seeds
def aggregateMetrics(metricsList):
    """
    Compute mean and standard deviation for each metric.

    Parameters:
    - metricsList : Evaluation metrics from different seeds
        
    Returns:
    - stats : Dictionary containing (mean, std) for each metric
        
    """

    stats = {}

    # Process each metric separately
    for key in metricsList[0].keys():
        values = np.array([m[key] for m in metricsList])

        # Store mean and standard deviation
        stats[key] = (float(values.mean()), float(values.std()))

    return stats



# Path visualisation
def plotPath(env, path, modelName):
    """
    Plot the greedy path learned by the agent on the warehouse grid.

    Parameters:
    - env : Environment instance
        
    - path : Sequence of visited positions
        
    - modelName : Name of algorithm for plot title and file name
    """

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, env.size, 1))
    ax.set_yticks(np.arange(-0.5, env.size, 1))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


    # Obstacles
    for (i, j) in env.staticObstacles:
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color="black"))


    # Agent path

    pathX = [pos[1] + 0.5 for pos in path]
    pathY = [pos[0] + 0.5 for pos in path]
    ax.plot(pathX, pathY, color="blue", linewidth=2, label="Path")


    # Draw special locations
    sx, sy = env.startPos
    ax.scatter(sy + 0.5, sx + 0.5, color="green", s=100, label="Start")

    px, py = env.pickupPos
    ax.scatter(py + 0.5, px + 0.5, color="orange", s=120, label="Pickup")

    dx, dy = env.deliveryPos
    ax.scatter(dy + 0.5, dx + 0.5, color="red", s=120, label="Delivery")

    cx, cy = env.chargerPos
    ax.scatter(cy + 0.5, cx + 0.5, color="purple", s=120, label="Charger")

    # §custom legend makes all visual elements explicit
    ax.set_title(f"{modelName.upper()} Agent Path", fontsize=14)
    legendElements = [
        plt.Line2D([0], [0], color="blue", lw=2, label="Path"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="Start"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="Pickup"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Delivery"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="purple", markersize=10, label="Charger"),
        Patch(facecolor="black", label="Obstacle")
    ]

    ax.legend(handles=legendElements, loc="upper left", bbox_to_anchor=(1, 1))

    # Set axis limits and invert y-axis for matrix-style view
    ax.set_xlim(0, env.size)
    ax.set_ylim(env.size, 0)

    # Save path visualisation
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/{modelName}_path.png", bbox_inches="tight")

    plt.show()



# Main pipeline
def main():
    """
    Run training, evaluation, comparison, and visualisation pipeline.
    """

    # Command-line arguments allow flexible execution
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["q_learning", "sarsa", "expected_sarsa", "all"],
        default="q_learning"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["episodes"])
    args = parser.parse_args()

    # Copy default configuration and update episode count
    config = dict(DEFAULT_CONFIG)
    config["episodes"] = args.episodes

    # Determine which models to run
    if args.model == "all":
        models = ["q_learning", "sarsa", "expected_sarsa"]
    else:
        models = [args.model]

    allResults = {}

    # Ensure result folders exist
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/json", exist_ok=True)

    # Train and Evaluate the models
    for modelName in models:

        internalModel = MODEL_MAP[modelName]
        print(f"\n Running model: {modelName}")

        perSeedMetrics = []

        for seed in args.seeds:
            print(f"Seed {seed} ...")

            # Train the selected model for one seed
            agent, rewards, metrics = trainOnce(internalModel, seed, config)
            perSeedMetrics.append(metrics)

            # Save learning curve plot
            plotFilePath = f"results/plots/{modelName}_seed{seed}.png"
            plotRewards(rewards, f"{modelName.replace('_', ' ').upper()} (seed={seed})", savePath=plotFilePath)

            # Save evaluation metrics for this seed
            with open(f"results/json/{modelName}_seed{seed}.json", "w") as file:
                json.dump(metrics, file, indent=2)

        # Aggregate results across all seeds
        stats = aggregateMetrics(perSeedMetrics)
        allResults[modelName] = stats


    # Compare models
    compareResults(allResults)


    # Generate path visualisations
    print("\n Generating Path Visualisations")

    for modelName in models:
        print(f"Generating path for: {modelName}")

        # Train once using fixed seed for path plotting
        internalModel = MODEL_MAP[modelName]
        agent, _, _ = trainOnce(internalModel, seed=28, config=config)

        env = WarehouseEnv(seed=42)
        path = env.getGreedyPath(agent.qTable)

        plotPath(env, path, modelName)

    # Save summary
    summary = {
        model: {
            metric: {"mean": value[0], "std": value[1]}
            for metric, value in stats.items()
        }
        for model, stats in allResults.items()
    }

    with open("results/json/summary.json", "w") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()