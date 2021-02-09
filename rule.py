import numpy as np
from observation import Observation
from action import Action


class Rule(object):
    def __init__(self, planet, minEntropy=1):
        self._minEntropy = minEntropy
        self.planet = planet
        self.state = np.zeros(
            (self.planet.height, self.planet.width), dtype=np.uint8)

    def updateState(self):
        for agent in self.agents:
            self.state[agent.h][agent.w] += 1

    def existAgent(self, h, w):
        H, W = self.state.shape
        if h >= 0 and h < H and w >= 0 and w < W:
            return self.state[h][w] > 0
        else:
            return False

    @property
    def agents(self):
        n = len(self._agents)
        for i, agent in enumerate(reversed(self._agents)):
            if agent.entropy < self._minEntropy:
                del self._agents[n-i-1]
        return self._agents

    @agents.setter
    def agents(self, value):
        for agent in value:
            self.born(agent)
        self._agents = value

    def born(self, agent, maxTry=1e6):
        for _ in range(int(maxTry)):
            h = np.random.randint(0, self.planet.height)
            w = np.random.randint(0, self.planet.width)
            if self.state[h][w] == 0:
                agent.h, agent.w = (h, w)
                self.state[h][w] += 1
                break

    def moveUp(self, agent):
        h, w = agent.h, agent.w
        if self.planet.inPlanet(h-1, w) and not self.existAgent(h-1, w):
            agent.h -= 1

    def moveDown(self, agent):
        h, w = agent.h, agent.w
        if self.planet.inPlanet(h+1, w) and not self.existAgent(h+1, w):
            agent.h += 1

    def moveLeft(self, agent):
        h, w = agent.h, agent.w
        if self.planet.inPlanet(h, w-1) and not self.existAgent(h, w-1):
            agent.w -= 1

    def moveRight(self, agent):
        h, w = agent.h, agent.w
        if self.planet.inPlanet(h, w+1) and not self.existAgent(h, w+1):
            agent.w += 1

    def observe(self, h, w):
        observations = []
        if self.planet.resourceAvilable(h, w):
            observations.append(Observation.Resources)
        if self.existAgent(h-1, w):
            observations.append(Observation.UpAgent)
        if self.existAgent(h+1, w):
            observations.append(Observation.DownAgent)
        if self.existAgent(h, w-1):
            observations.append(Observation.LeftAgent)
        if self.existAgent(h, w+1):
            observations.append(Observation.RightAgent)
        if h <= 0:
            observations.append(Observation.UpOut)
        if h >= self.planet.height - 1:
            observations.append(Observation.DownOut)
        if w <= 0:
            observations.append(Observation.LeftOut)
        if w >= self.planet.width - 1:
            observations.append(Observation.RightOut)
        return observations

    def tick(self):
        for agent in self.agents:
            h, w = agent.h, agent.w
            action = agent.forward(self.observe(h, w))
            if action == Action.MoveUp:
                self.moveUp(agent)
            if action == Action.MoveDown:
                self.moveDown(agent)
            if action == Action.MoveLeft:
                self.moveLeft(agent)
            if action == Action.MoveRight:
                self.moveRight(agent)
            if action == Action.Pick:
                agent.entropy += self.planet.decrease(h, w)
            if action == Action.Drop:
                if agent.entropy >= 1:
                    agent.entropy -= 1
                    self.planet.increase(h, w)
        self.updateState()
