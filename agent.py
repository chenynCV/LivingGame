import numpy as np
from collections import deque
from observation import Observation
from action import Action


class Agent(object):
    def __init__(self, entropy=100, historyLength=3):
        self.entropy = entropy
        self.ObservSpace = {x: i for i, x in enumerate(Observation)}
        self.ActionSpace = {x: i for i, x in enumerate(Action)}
        self.belief = np.zeros(
            (len(self.ObservSpace), len(self.ActionSpace)), dtype=np.int)
        self._observs = list(self.ObservSpace.keys())
        self._actions = list(self.ActionSpace.keys())
        self.history = self.initHistory(historyLength)
        self.age = 0

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

    def encodeObserv(self, observ):
        x = np.zeros((1, len(self.ObservSpace)))
        for item in observ:
            x[0, self.ObservSpace[item]] = 1
        return x

    def forward(self, observ):
        self.age += 1
        x = self.encodeObserv(observ)
        prob = x @ self.belief
        prob = np.clip(prob, 1e-6, 1e6).reshape(-1)
        prob = prob/np.sum(prob)
        action = np.random.choice(self._actions, p=prob)
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
