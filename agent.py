import numpy as np

class Agent(object):
    def __init__(self, space):
        self.life = 1
        self.belief = None
        self.pos = self.init(space)

    def init(self, space):
        N = space.size[0]
        pos = np.random.randint(0, N, size=2)
        return pos

    def forward(self, x):
        pass

    def backward(self):
        pass
 