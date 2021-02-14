import numpy as np
from collections import deque
from observation import Observation
from action import Action
from config import config


class Agent(object):
    def __init__(self, talent=0.9, historyLength=3, dtype=np.float):
        self.dtype = dtype
        self.ObservSpace = Observation()
        self.ActionSpace = Action()
        self.age = 0
        self.entropy = config.entropyBorn
        self.belief = self.initBelief()
        self.beliefGrad = np.zeros_like(self.belief)
        self.history = self.initHistory(historyLength)
        self.talent = talent
        self.historyLength = historyLength

    def initBelief(self, random=False):
        H = len(self.ObservSpace) + 1
        W = len(self.ActionSpace)
        if random:
            belief = np.array(np.random.randint(
                0, 2, size=(H, W)), dtype=self.dtype)
        else:
            belief = np.array(np.zeros((H, W)), dtype=self.dtype)
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

    def forward(self, observ):
        self.age += 1
        mask = np.full(len(self.ObservSpace)+1, True, dtype=bool)
        mask[1:] = (observ.reshape(-1) > 0)
        prob = np.zeros(len(self.ActionSpace))
        for i in range(len(self.ActionSpace)):
            prob[i] = np.sum(self.belief[mask, i])
        idx = np.argmax(prob)
        action = self.ActionSpace[idx]
        self.updateHistory((self.entropy, mask, action))
        return action

    def backward(self, optimalAction=None):
        grad = np.zeros_like(self.belief)
        entropyNow = self.entropy
        entropyLast, maskLast, actionLast = self.history[-1]
        if entropyLast <= entropyNow:
            grad[maskLast, actionLast] = 1
        else:
            grad[maskLast, actionLast] = -1
            if optimalAction is not None:
                grad[maskLast, optimalAction] = 1
        self.beliefGrad = self.talent * \
            self.beliefGrad + (1-self.talent)*grad
        self.belief += self.beliefGrad
        if np.max(self.belief) > 1e12:
            print('\nbelife shrink!\n')
            self.belief /= 2
        if np.min(self.belief) < 0:
            self.belief -= np.min(self.belief)
        self.updateHistory()

    def refresh(self, belief):
        self.age = 0
        self.entropy = config.entropyBorn
        self.belief = belief
        self.beliefGrad = np.zeros_like(self.belief)
        self.history = self.initHistory(self.historyLength)


if __name__ == '__main__':
    from config import config
    agent = Agent()
    x = np.random.random((1, config.width*config.height*config.bits))
    action = agent.forward(x)
    agent.backward()
    print(action)
