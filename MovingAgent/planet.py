import numpy as np


class Planet(object):
    def __init__(self, Resources=1e9, N=10):
        state = np.random.rand(N, N) > 0.6
        state = Resources*state/np.sum(state, axis=None)
        self.state = np.array(state, dtype=np.int)

    @property
    def width(self):
        return self.state.shape[1]

    @property
    def height(self):
        return self.state.shape[0]

    def inPlanet(self, h, w):
        if h >= 0 and h < self.height and w >= 0 and w < self.width:
            return True
        else:
            return False

    def resourceAvilable(self, h, w, v=1):
        if self.inPlanet(h, w):
            return self.state[h][w] >= v
        else:
            return False

    def decrease(self, h, w, v=1):
        if self.state[h][w] >= v:
            self.state[h][w] -= v
            return v
        else:
            return 0

    def increase(self, h, w, v=1):
        self.state[h][w] += v
        return self
