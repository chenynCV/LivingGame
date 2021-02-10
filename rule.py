import numpy as np
import uuid
from observation import Observation
from action import Action
from render import Render


class Rule(object):
    state = {}

    def __init__(self, planet, agents, viz=True):
        self._minEntropy = 1
        self._invEntropy = 3
        self.viz = viz
        self.planet = planet
        self.agents = [self.addAgent(agent) for agent in agents]
        self._agent = None
        if viz:
            self.renderObj = Render()

    def addAgent(self, agent):
        h = np.random.randint(0, self.planet.height)
        w = np.random.randint(0, self.planet.width)
        agent.h, agent.w = (h, w)
        agent.age = 0
        agent._ID = str(uuid.uuid4())
        if (h, w) not in self.state:
            self.state[(h, w)] = [agent._ID]
        else:
            self.state[(h, w)].append(agent._ID)
        return agent

    def existAgent(self, h, w):
        return (h, w) in self.state and len(self.state[(h, w)]) > 0

    def moveUp(self, agent):
        h, w = agent.h, agent.w
        if self.planet.inPlanet(h-1, w):
            agent.h -= 1

    def moveDown(self, agent):
        h, w = agent.h, agent.w
        if self.planet.inPlanet(h+1, w):
            agent.h += 1

    def moveLeft(self, agent):
        h, w = agent.h, agent.w
        if self.planet.inPlanet(h, w-1):
            agent.w -= 1

    def moveRight(self, agent):
        h, w = agent.h, agent.w
        if self.planet.inPlanet(h, w+1):
            agent.w += 1

    def observe(self, h, w):
        observations = []
        if self.planet.resourceAvilable(h, w, self._invEntropy):
            observations.append(Observation.Resource)
        if self.planet.resourceAvilable(h-1, w, self._invEntropy):
            observations.append(Observation.ResourceUp)
        if self.planet.resourceAvilable(h+1, w, self._invEntropy):
            observations.append(Observation.ResourceDown)
        if self.planet.resourceAvilable(h, w-1, self._invEntropy):
            observations.append(Observation.ResourceLeft)
        if self.planet.resourceAvilable(h, w+1, self._invEntropy):
            observations.append(Observation.ResourceRight)
        if self.existAgent(h-1, w):
            observations.append(Observation.AgentUp)
        if self.existAgent(h+1, w):
            observations.append(Observation.AgentDown)
        if self.existAgent(h, w-1):
            observations.append(Observation.AgentLeft)
        if self.existAgent(h, w+1):
            observations.append(Observation.AgentRight)
        if h <= 0:
            observations.append(Observation.OutUp)
        if h >= self.planet.height - 1:
            observations.append(Observation.OutDown)
        if w <= 0:
            observations.append(Observation.OutLeft)
        if w >= self.planet.width - 1:
            observations.append(Observation.OutRight)
        return observations

    def tick(self):
        n = len(self.agents)
        if n == 1:
            self._agent = self.agents[0]
        for i, agent in enumerate(reversed(self.agents)):
            agent.age += 1
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

            # update state
            self.state[(h, w)].remove(agent._ID)
            if (agent.h, agent.w) in self.state:
                self.state[(agent.h, agent.w)].append(agent._ID)
            else:
                self.state[(agent.h, agent.w)] = [agent._ID]

            # update agent
            agent.entropy -= 1
            if self.planet.state[h][w] >= self._invEntropy:
                self.planet.state[h][w] -= self._invEntropy
                agent.entropy += self._invEntropy
            if agent.entropy < self._minEntropy:
                del self.agents[n-i-1]

        if self.viz:
            self.renderObj.update(self.planet, self.agents)
