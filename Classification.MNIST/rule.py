import uuid
import numpy as np
from render import Render
from agent import Agent
from config import config


class Rule(object):

    def __init__(self, planet, agents):
        self._minEntropy = 1
        self._invEntropy = 3
        self._periodEntropy = 1
        self.eraseProb = 0.0001
        self.timeline = 0
        self.renderObj = None
        self.planet = planet
        self.agentNum = len(agents)
        for agent in agents:
            self.addAgent(agent)

    def addAgent(self, agent):
        if not hasattr(self, 'agents'):
            self.agents = []
        agent._ID = str(uuid.uuid4())
        agent.acc = 0
        self.agents.append(agent)
        return self

    def observe(self):
        observations = self.planet.fetch()
        return observations

    def moveAverage(self, m, v, period=100*config.classNum):
        m = (m*(period-1)+v)/period
        return m

    def update(self, agent, action, optimalAction):
        agent.entropy -= self._periodEntropy
        agent.action = action
        reward = (action == optimalAction)
        if reward:
            agent.entropy += self._invEntropy
            agent.acc = self.moveAverage(agent.acc, 1)
        else:
            agent.acc = self.moveAverage(agent.acc, 0)

    def hybrid(self, agentA, agentB):
        agent = Agent()
        agent.belief = (agentA.belief + agentB.belief)/2
        return agent

    def sortAgents(self):
        def _key(x):
            return x.entropy
        self.agents.sort(key=_key, reverse=True)
        return self.agents

    def tick(self, viz=False):
        self.timeline += 1
        n = len(self.agents)
        for i, agent in enumerate(reversed(self.agents)):
            observ, optimalAction = self.observe()
            action = agent.forward(observ)
            self.update(agent, action, optimalAction)
            agent.backward(optimalAction)

            # erase dead
            if agent.entropy < self._minEntropy:
                del self.agents[n-i-1]

        # sort according to entropy
        self.sortAgents()

        # hybrid or erase
        if len(self.agents) < self.agentNum:
            self.addAgent(self.hybrid(self.agents[0], self.agents[1]))
        elif len(self.agents) > self.agentNum:
            del self.agents[-1]
        elif np.random.rand() > 1 - self.eraseProb:
            idx = np.random.randint(0, len(self.agents))
            del self.agents[idx]

        if viz and len(self.agents) > 0:
            if self.renderObj is None:
                self.renderObj = Render()
            self.renderObj.update(self.agents[0])
