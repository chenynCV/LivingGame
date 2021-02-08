import numpy as np


class Universe(object):
    def __init__(self, Resources=1e9, N=10):
        self.state = self.init(Resources, N)

    def init(self, Resources, N):
        state = np.random.rand(N, N)
        state *= Resources/np.sum(state, axis=None)
        return state
