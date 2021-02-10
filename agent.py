import numpy as np
from observation import Observation
from action import Action


class Agent(object):
    def __init__(self, entropy=100, priorConfidence=1):
        self.entropy = entropy
        self.ObservSpace = {x: i for i, x in enumerate(Observation)}
        self.ActionSpace = {x: i for i, x in enumerate(Action)}
        self.belief = self.priorBelief(
            self.initBelief(), confidence=priorConfidence)
        self._observs = list(self.ObservSpace.keys())
        self._actions = list(self.ActionSpace.keys())

    def initBelief(self):
        a = np.random.rand()
        b = np.random.rand()
        belief = np.random.beta(a, b, size=(
            len(self.ObservSpace), len(self.ActionSpace)))
        belief = np.clip(belief, 1e-3, 1e3)
        belief /= np.sum(belief)
        return belief

    def priorBelief(self, belief, confidence=1):
        belief[self.ObservSpace[Observation.Resource]
               ][self.ActionSpace[Action.Nothing]] *= confidence
        belief[self.ObservSpace[Observation.ResourceUp]
               ][self.ActionSpace[Action.MoveUp]] *= confidence
        belief[self.ObservSpace[Observation.ResourceDown]
               ][self.ActionSpace[Action.MoveDown]] *= confidence
        belief[self.ObservSpace[Observation.ResourceLeft]
               ][self.ActionSpace[Action.MoveLeft]] *= confidence
        belief[self.ObservSpace[Observation.ResourceRight]
               ][self.ActionSpace[Action.MoveRight]] *= confidence
        return belief

    def forward(self, x):
        observ = np.zeros((1, len(self.ObservSpace)))
        for item in x:
            observ[0, self.ObservSpace[item]] = 1
        prob = observ @ self.belief
        prob = np.clip(prob, 1e-3, 1e3).reshape(-1)
        prob = prob/np.sum(prob)
        action = np.random.choice(self._actions, p=prob)
        return action

    def backward(self):
        pass


if __name__ == '__main__':
    agent = Agent()
    x = [Observation.Resource]
    action = agent.forward(x)
    print(action)
