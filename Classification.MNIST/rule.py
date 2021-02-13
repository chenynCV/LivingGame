import uuid
import numpy as np
from render import Render
from config import config


class Rule(object):

    def __init__(self, planet, agents, viz=True):
        self._minEntropy = 1
        self._invEntropy = 1
        self._periodEntropy = 1
        self.viz = viz
        self.planet = planet
        self.agents = [self.addAgent(agent) for agent in agents]
        self.winAgent = self.agents[0]
        self.timeline = 0
        if viz:
            self.renderObj = Render()

    def addAgent(self, agent):
        agent._ID = str(uuid.uuid4())
        agent.acc = 0
        return agent

    def observe(self):
        observations = self.planet.fetch()
        return observations

    def moveAverage(self, m, v, period=100*config.classNum):
        m = (m*(period-1)+v)/period
        return m

    def update(self, agent, action, optimalAction):
        agent.action = action
        reward = (action == optimalAction)
        if reward:
            agent.entropy += self._invEntropy
            agent.acc = self.moveAverage(agent.acc, 1)
        else:
            agent.entropy -= self._periodEntropy
            agent.acc = self.moveAverage(agent.acc, 0)

    def hybridMutation(self, p=0.5):
        for agent in self.agents:
            if np.random.rand() > p:
                agent.belief = np.right_shift(
                    agent.belief + self.winAgent.belief, 1)

    def tick(self):
        self.timeline += 1
        n = len(self.agents)
        for i, agent in enumerate(reversed(self.agents)):
            observ, optimalAction = self.observe()
            action = agent.forward(observ)
            self.update(agent, action, optimalAction)
            agent.backward()

            if agent.entropy < self._minEntropy:
                del self.agents[n-i-1]
            elif agent.entropy >= self.winAgent.entropy:
                    self.winAgent = agent
        
        # hybrid
        if self.timeline > len(self.planet):
            self.hybridMutation(p=0.5)

        if self.viz:
            self.renderObj.update(self.agents)
