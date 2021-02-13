import numpy as np


def arrayToBits(array, n=8, dtype=np.int32):
    H, W = array.shape[:2]
    array = np.array(array, dtype=dtype)
    bits = np.zeros((H, W, n), dtype=dtype)
    for i in range(n):
        bits[:, :, i] = np.array(np.bitwise_and(array, 2**i) > 0, dtype=dtype)
    return bits


def bitsToarray(bits, dtype=np.int32):
    H, W, n = bits.shape
    bits = np.array(bits, dtype=dtype)
    array = np.zeros((H, W), dtype=dtype)
    for i in range(n):
        array += (2**i)*bits[:, :, i]
    return array
