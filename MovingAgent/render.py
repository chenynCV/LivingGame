import numpy as np
import matplotlib.pyplot as plt


class Render(object):

    def __init__(self, fullScreen=True):
        self.fig, self.ax = plt.subplots(1, 2)
        if fullScreen:
            self.fig.canvas.manager.full_screen_toggle()
        self.fig.suptitle('Living Game', fontsize=32)
        self.ax[0].axis('off')
        self.ax[1].axis('off')

    def plotPlanet(self, ax, planet, agents=None):
        ax.cla()
        ax.set_title('Evolution')

        # Plot the heatmap
        im = ax.imshow(planet.state, cmap='BuGn')

        # render agent
        if agents is not None:
            for agent in agents:
                im.axes.text(agent.w, agent.h, agent.entropy,
                             horizontalalignment="center", verticalalignment="center")

    def plotAgentBelief(self, ax, agent):
        ax.cla()
        ax.set_title("The winner's belief")

        # Plot the heatmap
        im = ax.imshow(agent.belief, cmap="Oranges")

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(agent.actions)))
        ax.set_yticks(np.arange(len(agent.observs)))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(agent.actions)
        ax.set_yticklabels(agent.observs)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        for h in range(agent.belief.shape[0]):
            for w in range(agent.belief.shape[1]):
                im.axes.text(w, h, agent.belief[h, w],
                             horizontalalignment="center", verticalalignment="center")

    def update(self, planet, agents, interval=0.01):
        if len(agents) > 0:
            self.plotPlanet(self.ax[0], planet, agents)

            maxEntropy = 0
            winnerAgent = None
            for agent in agents:
                if agent.entropy > maxEntropy:
                    winnerAgent = agent
                    maxEntropy = agent.entropy
            self.plotAgentBelief(self.ax[1], winnerAgent)
        plt.pause(interval)

    def __del__(self):
        plt.close(self.fig)
