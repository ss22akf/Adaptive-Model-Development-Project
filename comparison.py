import os
import matplotlib.pyplot as plt


def compareResults(results):
    """
    Compare performance of all 3 RL models. This function:
        - Prints evaluation metrics
        - Prepares data for plotting
        - Ensures output directory exists
    """

    # PRINT RESULTS
    print("\n Comparison (Mean ± Std)")

    # Loop through each model and display metrics
    for name, stats in results.items():
        print(f"\n{name}")
        for k, (mean, std) in stats.items():
            # Format output to 3 decimal places
            print(f"{k}: {mean:.3f} ± {std:.3f}")

    # List of model names (q_learning, sarsa, expected_sarsa)
    models = list(results.keys())

    # Extract mean values for each metric
    # Index [0] corresponds to mean (mean, std)
    success = [results[m]["successRate"][0] for m in models]
    steps = [results[m]["avgSteps"][0] for m in models]
    collisions = [results[m]["collisions"][0] for m in models]

    # Create output directory
    os.makedirs("results/plots", exist_ok=True)

    # Success Rate
    plt.figure()
    plt.bar(models, success)
    plt.title("Success Rate Comparison")
    plt.ylabel("Success Rate")
    plt.savefig("results/plots/success_comparison.png", bbox_inches="tight")
    plt.show()

    # Steps
    plt.figure()
    plt.bar(models, steps)
    plt.title("Average Steps Comparison")
    plt.ylabel("Steps")
    plt.savefig("results/plots/steps_comparison.png", bbox_inches="tight")
    plt.show()

    # Collisions
    plt.figure()
    plt.bar(models, collisions)
    plt.title("Collision Comparison")
    plt.ylabel("Collisions")
    plt.savefig("results/plots/collision_comparison.png", bbox_inches="tight")
    plt.show()