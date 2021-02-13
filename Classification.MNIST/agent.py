import numpy as np
from collections import deque
from observation import Observation
from action import Action
from config import config


class Agent(object):
    def __init__(self, entropy=100, historyLength=3, dtype=np.int64):
        self.entropy = entropy
        self.dtype = dtype
        self.ObservSpace = Observation()
        self.ActionSpace = Action()
        self.belief = self.initBelief()
        self.history = self.initHistory(historyLength)
        self.age = 0

    def initBelief(self):
        H = len(self.ObservSpace) + 1
        W = len(self.ActionSpace)
        belief = np.random.randint(0, 2, size=(H, W), dtype=self.dtype)
        return belief

    def initHistory(self, length):
        self.historyLength = length
        history = deque()
        for _ in range(length):
            history.append((self.entropy, None, None, None))
        return history

    def updateHistory(self, *wargs):
        if len(self.history) > self.historyLength:
            return self.history.popleft()
        else:
            return self.history.append(*wargs)

    def forward(self, observ):
        self.age += 1
        mask = np.full(len(self.ObservSpace)+1, True, dtype=bool)
        mask[1:] = (observ.reshape(-1) > 0)
        prob = np.zeros(len(self.ActionSpace))
        for i in range(len(self.ActionSpace)):
            prob[i] = np.sum(self.belief[mask, i])
        idx = np.argmax(prob)
        action = self.ActionSpace[idx]
        self.updateHistory((self.entropy, mask, prob, action))
        return action

    def backward(self, optimalAction=None):
        lastEntropy, lastMask, lastProb, lastAction = self.history[-1]
        if lastEntropy <= self.entropy:
            self.belief[lastMask, lastAction] += 1
        else:
            self.belief[lastMask, lastAction] -= 1
            if optimalAction is not None:
                self.belief[lastMask, optimalAction] += self.dtype(
                    lastProb.max() - lastProb[lastAction] + 1)
        if np.max(self.belief) > 1e12:
            print('\nbelife shrink!\n')
            self.belief = np.right_shift(self.belief, 1)
        if np.min(self.belief) < 0:
            self.belief -= np.min(self.belief)
        self.updateHistory()


if __name__ == '__main__':
    from config import config
    agent = Agent()
    x = np.random.random((1, config.width*config.height*config.bits))
    action = agent.forward(x)
    agent.backward()
    print(action)
