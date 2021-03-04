import numpy as np
from config import config


class Observation(object):

    Space = np.arange(0, config.width*config.height*config.bits)

    def __getitem__(self, idx):
        return self.Space[idx]

    def __len__(self):
        return len(self.Space)

    def __iter__(self):
        for i in range(0, len(self)):
            yield self.Space[i]
