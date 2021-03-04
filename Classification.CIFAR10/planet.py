import numpy as np
import pickle as pkl

from numpy import random
import utils
from config import config


class Planet(object):
    def __init__(self, dataPath=config.dataPath, classNum=config.classNum, shuffle=False):
        self.classNum = classNum
        self.data = self.load(dataPath)
        if shuffle:
            self.shuffle()

    def load(self, dataPath):
        with open(dataPath, 'rb') as f:
            data = pkl.load(f, encoding='bytes')
        X = data[b'data']
        X = [np.reshape(x, (3, 32, 32)).transpose(1, 2, 0) for x in X]
        Y = data[b'labels']
        return np.array(X), np.array(Y)

    def shuffle(self):
        index = np.random.permutation(len(self))
        X = self.data[0][index]
        Y = self.data[1][index]
        return X, Y

    def fetch(self):
        idx = np.random.randint(0, len(self))
        x, y = self[idx]
        return utils.arrayToBits(x, config.bits), y

    def __getitem__(self, index):
        x = self.data[0][index]
        y = self.data[1][index]
        return x, y

    def __len__(self):
        return len(self.data[0])

    def __iter__(self):
        for i in range(0, len(self)):
            yield self.data[0][i], self.data[1][i]


if __name__ == '__main__':
    import cv2

    dataset = Planet()
    for _ in range(100):
        x, y = dataset.fetch()
        # x = utils.bitsToarray(x)
        cv2.imshow('img', np.array(x, dtype=np.uint8))
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
