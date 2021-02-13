import numpy as np
import pickle as pkl
import gzip
import utils
from config import config


class Planet(object):
    def __init__(self, dataPath=config.dataPath, classNum=config.classNum, split='train', shuffle=False):
        self.classNum = classNum
        self.split = split
        self.data = self.load(dataPath)
        if shuffle:
            self.shuffle()

    def load(self, dataPath):
        with gzip.open(dataPath, 'rb') as f:
            trainData, valData, testData = pkl.load(f, encoding='bytes')
        if self.split == 'train':
            X = [np.reshape(x, (28, 28)) for x in trainData[0]]
            Y = trainData[1]
        elif self.split == 'val':
            X = [np.reshape(x, (28, 28)) for x in valData[0]]
            Y = valData[1]
        elif self.split == 'test':
            X = [np.reshape(x, (28, 28)) for x in testData[0]]
            Y = testData[1]
        else:
            raise NotImplementedError
        return np.array(X), np.array(Y)

    def shuffle(self):
        index = np.random.permutation(len(self))
        X = self.data[0][index]
        Y = self.data[1][index]
        return X, Y

    def fetch(self):
        idx = np.random.randint(0, len(self))
        x, y = self[idx]
        x = np.array(255*x, dtype=np.uint8)
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
        x = utils.bitsToarray(x)
        cv2.imshow('img', np.array(x, dtype=np.uint8))
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
