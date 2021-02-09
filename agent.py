import numpy as np
from observation import Observation
from action import Action


class Agent(object):
    def __init__(self, entropy=10):
        self.entropy = entropy
        self.avilableObservation = list(Observation)
        self.avilableAction = list(Action)
        self.belief = self.initBelief()

    def initBelief(self):
        belief = np.random.rand(
            len(self.avilableObservation), len(self.avilableAction))
        return belief

    def forward(self, x):
        O = np.zeros((1, len(self.avilableObservation)))
        for i, observ in enumerate(self.avilableObservation):
            if observ in x:
                O[0, i] = 1
        A = np.clip(O @ self.belief, 1e-10, 1e10)
        A = A.reshape(-1)/np.sum(A)
        a = np.random.choice(self.avilableAction, p=A.reshape(-1))
        return a

    def backward(self):
        pass


if __name__ == '__main__':
    agent = Agent()
    x = [Observation.Resources]
    action = agent.forward(x)
    print(action)
