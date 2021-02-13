import uuid
from render import Render
from config import config


class Rule(object):

    def __init__(self, planet, agents, viz=True):
        self._minEntropy = 1
        self._invEntropy = 1
        self.viz = viz
        self.planet = planet
        self.agents = [self.addAgent(agent) for agent in agents]
        if viz:
            self.renderObj = Render()

    def addAgent(self, agent):
        agent._ID = str(uuid.uuid4())
        agent.acc = 0
        return agent

    def observe(self):
        observations = self.planet.fetch()
        return observations

    def moveAverage(self, m, v, period=config.classNum):
        m = (m*(period-1)+v)/period
        return m

    def update(self, agent, action, optimalAction):
        agent.action = action
        reward = (action == optimalAction)
        if reward:
            agent.entropy += self._invEntropy
            agent.acc = self.moveAverage(agent.acc, 1)
        else:
            agent.entropy -= 1
            agent.acc = self.moveAverage(agent.acc, 0)

    def tick(self):
        n = len(self.agents)
        for i, agent in enumerate(reversed(self.agents)):
            observ, optimalAction = self.observe()
            action = agent.forward(observ)
            self.update(agent, action, optimalAction)
            agent.backward(optimalAction)

            # erase
            if agent.entropy < self._minEntropy:
                del self.agents[n-i-1]

        if self.viz:
            self.renderObj.update(self.agents)
