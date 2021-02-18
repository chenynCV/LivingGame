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

    def initBelief(self):
        belief = np.ones((self.height, self.width, 2))
        belief = np.array(belief, dtype=self.dtype)
        return belief

    @property
    def width(self):
        return len(self.ActionSpace)

    @property
    def height(self):
        return len(self.ObservSpace) + 1

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
        prob = np.sum(self.belief[mask, :, :], axis=0)
        prob = prob[:, 0] / (prob[:, 1] + 1)
        index = np.argmax(prob)
        action = self.ActionSpace[index]

        # update history
        self.updateHistory((self.entropy, mask, action))
        return action

    def backward(self, optimalAction=None):
        entropyNow = self.entropy
        entropyLast, maskLast, actionLast = self.history[-1]
        grad = np.zeros_like(self.belief)
        grad[maskLast, actionLast, 1] = 1
        if entropyLast <= entropyNow:
            grad[maskLast, actionLast, 0] = 1
        if optimalAction is not None:
            grad[maskLast, optimalAction, 0] = 1
        self.beliefGrad = self.talent * \
            self.beliefGrad + (1-self.talent)*grad
        self.belief += self.beliefGrad
        self.updateHistory()


if __name__ == '__main__':
    from config import config
    agent = Agent()
    x = np.random.random((1, config.width*config.height*config.bits))
    action = agent.forward(x)
    agent.backward()
    print(action)
