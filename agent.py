import numpy as np
from collections import deque, OrderedDict
from observation import Observation
from action import Action


class Agent(object):
    def __init__(self, entropy=100, historyLength=3, p=0.2):
        self.entropy = entropy
        self.ObservSpace = self.initObserv()
        self.ActionSpace = self.initAction(p)
        self.belief = np.zeros(
            (len(self.ObservSpace), len(self.ActionSpace)), dtype=np.int)
        self.history = self.initHistory(historyLength)
        self.age = 0

    def initObserv(self):
        ObservSpace = OrderedDict()
        for i, x in enumerate(Observation):
            ObservSpace[x] = i
        return ObservSpace

    def initAction(self, p=0.2):
        ActionSpace = OrderedDict()
        for i, x in enumerate(Action):
            ActionSpace[x] = i
        if np.random.rand() > p:
            ActionSpace.popitem()
        else:
            self.entropy = self.entropy // 2
        return ActionSpace

    @property
    def observs(self):
        return list(self.ObservSpace.keys())

    @property
    def actions(self):
        return list(self.ActionSpace.keys())

    def addObserv(self, observs):
        for observ in observs:
            v = len(self.ObservSpace)
            self.ObservSpace[observ] = v
            self.belief = np.append(self.belief, np.zeros(
                (1, len(self.ActionSpace)), dtype=np.int), axis=0)

    def addAction(self, actions):
        for action in actions:
            v = len(self.ActionSpace)
            self.ActionSpace[action] = v
            self.belief = np.append(self.belief, np.zeros(
                (len(self.ObservSpace), 1), dtype=np.int), axis=1)

    def initHistory(self, length):
        self.historyLength = length
        history = deque()
        for _ in range(length):
            history.append((self.entropy, Observation.Prior, Action.Nothing))
        return history

    def updateHistory(self, *wargs):
        if len(self.history) > self.historyLength:
            return self.history.popleft()
        else:
            return self.history.append(*wargs)

    def encodeObserv(self, observs):
        x = np.zeros((1, len(self.ObservSpace)))
        for observ in observs:
            x[0, self.ObservSpace[observ]] = 1
        return x

    def forward(self, observ):
        self.age += 1
        x = self.encodeObserv(observ)
        prob = x @ self.belief
        prob = np.clip(prob, 1e-6, 1e6).reshape(-1)
        prob = prob/np.sum(prob)
        action = np.random.choice(self.actions, p=prob)
        self.updateHistory((self.entropy, observ, action))
        return action

    def backward(self):
        lastEntropy, lastObserv, lastAction = self.history[-1]
        if lastEntropy <= self.entropy:
            for item in lastObserv:
                self.belief[self.ObservSpace[item],
                            self.ActionSpace[lastAction]] += 1
        else:
            for item in lastObserv:
                self.belief[self.ObservSpace[item],
                            self.ActionSpace[lastAction]] -= 1
        if np.min(self.belief) < 0:
            self.belief -= np.min(self.belief)
        self.updateHistory()


if __name__ == '__main__':
    agent = Agent()
    x = [Observation.Resource]
    action = agent.forward(x)
    agent.backward()
    print(action)
