import numpy as np
import matplotlib.pyplot as plt


class Render(object):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.cbar = None

    def update(self, planet, agents, interval=0.001):
        self.ax.cla()
        self.ax.axis('off')

        # Plot the heatmap
        im = self.ax.imshow(planet.state, cmap="BuGn")

        # Create colorbar
        if self.cbar is None:
            self.cbar = self.ax.figure.colorbar(im, ax=self.ax)
            self.cbar.ax.set_ylabel('Resources', rotation=-90, va="bottom")

        # render agent
        for agent in agents:
            im.axes.text(agent.w, agent.h, agent.entropy,
                         horizontalalignment="center", verticalalignment="center")

        plt.pause(interval)

def plotAgentBelief(agent):
    fig, ax = plt.subplots()

    # Plot the heatmap
    im = ax.imshow(agent.belief, cmap="Greys")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Resources', rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(agent.avilableAction)))
    ax.set_yticks(np.arange(len(agent.avilableObservation)))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(agent.avilableAction)
    ax.set_yticklabels(agent.avilableObservation)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    ax.set_title("The winner's belief")
    fig.tight_layout()
    plt.show()
