import os
import matplotlib.pyplot as plt

def plotRewards(rewards, title, savePath=None):
    """
    Plot reward progression over training episodes.

    Parameters:
    - rewards : 
        Total reward obtained in each episode
    - title : 
        Title of the plot (e.g., model name and seed)
    - savePath : 
        File path to save the plot image
    """

    # Create a new figure for the plot
    plt.figure()

    # Plot reward values across episodes
    plt.plot(rewards)

    # Add labels and title
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    # Display grid for better readability
    plt.grid()

    # Save plot
    if savePath:
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(savePath), exist_ok=True)

        # Save the plot as an image file
        plt.savefig(savePath, bbox_inches='tight')

    # Display the plot
    plt.show()