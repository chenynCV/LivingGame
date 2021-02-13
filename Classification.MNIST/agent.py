import numpy as np
from collections import deque
from observation import Observation
from action import Action
from config import config


class Agent(object):
    def __init__(self, entropy=100, talent=0.99, historyLength=3, dtype=np.int64):
        self.entropy = entropy
        self.talent = talent
        self.dtype = dtype
        self.ObservSpace = Observation()
        self.ActionSpace = Action()
        self.belief = self.initBelief()
        self.history = self.initHistory(historyLength)
        self.age = 0

    def initBelief(self):
        H = len(self.ObservSpace) + 1
        W = len(self.ActionSpace)
        belief = np.random.randint(
            0, config.classNum, size=(H, W), dtype=self.dtype)
        return belief

    def initHistory(self, length):
        self.historyLength = length
        history = deque()
        for _ in range(length):
            history.append((self.entropy, None, None))
        return history

    def updateHistory(self, *wargs):
        if len(self.history) > self.historyLength:
            return self.history.popleft()
        else:
            return self.history.append(*wargs)

    def encodeObserv(self, observ):
        x = np.zeros((1, 1 + len(self.ObservSpace)), dtype=self.dtype)
        x[0, 0] = 1
        x[0, 1:] = observ.reshape(1, -1)
        return x

    def forward(self, observ):
        self.age += 1
        observ = self.encodeObserv(self.dtype(observ))
        prob = observ @ self.belief
        prob = prob.reshape(-1)
        idx = np.argmax(prob)
        action = self.ActionSpace[idx]
        self.updateHistory((self.entropy, observ, action))
        return action

    def backward(self, optimalAction=None):
        lastEntropy, lastObserv, lastAction = self.history[-1]
        mask = lastObserv.reshape(-1) > 0
        if lastEntropy < self.entropy:
            self.belief[mask, lastAction] += 1
        elif optimalAction is not None:
            self.belief[mask, optimalAction] += 1
        if np.max(self.belief) > 1e12:
            self.belief = np.array(self.belief*self.talent, dtype=self.dtype)
        self.updateHistory()


if __name__ == '__main__':
    from config import config
    agent = Agent()
    x = np.random.random((1, config.width*config.height*config.bits))
    action = agent.forward(x)
    agent.backward()
    print(action)
