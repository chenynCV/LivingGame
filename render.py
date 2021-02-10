import numpy as np
import matplotlib.pyplot as plt


class Render(object):

    def __init__(self, title='Evolution', label='Resources', cmap='BuGn'):
        self.title = title
        self.label = label
        self.cmap = cmap
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.full_screen_toggle()
        self.ax.axis('off')
        self.cbar = None

    def update(self, planet, agents, interval=0.001):
        self.ax.cla()
        self.ax.set_title(self.title)

        # Plot the heatmap
        im = self.ax.imshow(planet.state, cmap=self.cmap)

        # Create colorbar
        if self.cbar is None:
            self.cbar = self.ax.figure.colorbar(im, ax=self.ax)
            self.cbar.ax.set_ylabel(self.label, rotation=-90, va="bottom")

        # render agent
        for agent in agents:
            im.axes.text(agent.w, agent.h, agent.entropy,
                         horizontalalignment="center", verticalalignment="center")

        plt.pause(interval)

    def __del__(self):
        plt.close(self.fig)


def plotAgentBelief(agent):
    fig, ax = plt.subplots()
    fig.canvas.manager.full_screen_toggle()

    # Plot the heatmap
    im = ax.imshow(agent.belief, cmap="Greys")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Resources', rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(agent._actions)))
    ax.set_yticks(np.arange(len(agent._observs)))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(agent._actions)
    ax.set_yticklabels(agent._observs)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax.set_title("The winner's belief")
    fig.tight_layout()
    plt.show()
