import uuid
import numpy as np
from render import Render
from config import config


class Rule(object):

    def __init__(self, planet, agents):
        self._minEntropy = 1
        self._invEntropy = 0.5
        self._periodEntropy = 0.4
        self.planet = planet
        self.agents = [self.addAgent(agent) for agent in agents]
        self.winAgent = self.agents[0]
        self.hybridProp = 100 / len(self.planet)
        self.timeline = 0
        self.renderObj = None

    def addAgent(self, agent):
        agent._ID = str(uuid.uuid4())
        agent.acc = np.zeros(len(agent.ActionSpace))
        return agent

    def observe(self):
        observations = self.planet.fetch()
        return observations

    def moveAverage(self, m, v, period=3*config.classNum):
        m = (m*(period-1)+v)/period
        return m

    def update(self, agent, action, optimalAction):
        agent.entropy -= self._periodEntropy
        agent.action = action
        reward = (action == optimalAction)
        if reward:
            agent.entropy += self._invEntropy
            agent.acc[action] = self.moveAverage(agent.acc[action], 1)
        else:
            agent.acc[action] = self.moveAverage(agent.acc[action], 0)

    def hybridMutation(self, idx):
        if self.agents[idx].entropy < self.winAgent.entropy and \
                self.agents[idx].age > 1/self.hybridProp and \
                self.winAgent.age > 1/self.hybridProp:
            ageDiff = abs(self.agents[idx].age - self.winAgent.age)
            if ageDiff*self.hybridProp < abs(np.random.normal()):
                H, W = self.agents[idx].belief.shape
                belief = np.where(np.random.rand(H, W) > 0.5,
                                  self.agents[idx].belief, self.winAgent.belief)
                self.agents[idx].refresh(
                    belief / np.sum(belief) * H * W / self.hybridProp)
                self.agents[idx].acc = (
                    self.agents[idx].acc + self.winAgent.acc) / 2
                print("hybrid!")

    def tick(self, viz=False):
        self.timeline += 1
        n = len(self.agents)
        for i, agent in enumerate(reversed(self.agents)):
            observ, optimalAction = self.observe()
            action = agent.forward(observ)
            self.update(agent, action, optimalAction)
            agent.backward(optimalAction)

            if agent.entropy < self._minEntropy:
                del self.agents[n-i-1]
            elif agent.entropy >= self.winAgent.entropy:
                self.winAgent = agent
            else:
                # hybrid
                if np.random.rand() > 1 - self.hybridProp:
                    self.hybridMutation(i)

        if viz:
            if self.renderObj is None:
                self.renderObj = Render()
            self.renderObj.update(self.agents)
