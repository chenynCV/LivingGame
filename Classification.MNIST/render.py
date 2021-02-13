import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import utils
from config import config


class Render(object):

    def __init__(self, fullScreen=False):
        self.fig = plt.figure(constrained_layout=True, figsize=[9.6, 7.2])
        gs = GridSpec(4, 4, figure=self.fig)
        self.axObserv = self.fig.add_subplot(gs[:, 0])
        self.axBelief = []
        for r in range(0, 4):
            for c in range(1, 4):
                self.axBelief.append(self.fig.add_subplot(gs[r, c]))

        self.fig.suptitle('Living Game', fontsize=32)
        if fullScreen:
            self.fig.canvas.manager.full_screen_toggle()

    def plotObserv(self, ax, agent, title='Observation'):
        observation = agent.history[-1][1]
        observation = observation[1:].reshape(
            config.width, config.height, config.bits)
        observation = utils.bitsToarray(observation)
        ax.cla()
        ax.set_title(title)
        ax.imshow(observation, cmap='BuGn')

    def plotBelief(self, axs, agent):
        ax = axs[0]
        ax.cla()
        ax.set_title('Prior')
        ax.imshow(agent.belief[0:1, :], cmap="Oranges")
        ax.tick_params(labelleft=False)
        ax.set_xticks(np.arange(len(agent.ActionSpace)))
        ax.set_xticklabels(agent.ActionSpace)

        for i in range(1, len(axs)):
            ax = axs[i]
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.cla()
            if i-1 < agent.belief.shape[1]:
                ax.set_title('Digit ' + str(i-1))
                subBelief = agent.belief[1:, i-1]
                subBelief = subBelief.reshape(
                    config.width, config.height, config.bits)
                subBelief = utils.bitsToarray(subBelief)
                ax.imshow(subBelief, cmap="Oranges")

    def update(self, agents, interval=0.01):
        if len(agents) > 0:
            maxEntropy = 0
            winAgent = None
            for agent in agents:
                if agent.entropy > maxEntropy:
                    winAgent = agent
                    maxEntropy = agent.entropy
            title = 'E: {}, Action: {}, Acc: {:.2f}'.format(
                winAgent.entropy, winAgent.action, winAgent.acc)
            self.plotObserv(self.axObserv, winAgent, title=title)
            self.plotBelief(self.axBelief, winAgent)
        plt.pause(interval)

    def __del__(self):
        plt.close(self.fig)
