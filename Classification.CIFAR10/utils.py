import numpy as np


def arrayToBits(array, n=8, dtype=np.uint8):
    if len(array.shape) == 2: 
        H, W = array.shape[:2]
        C = 1
    elif len(array.shape) == 3:
        H, W, C = array.shape
    else:
        raise NotImplementedError
    array = np.array(array, dtype=dtype)
    bits = np.zeros((H, W, C*n), dtype=dtype)
    for i in range(n):
        bits[:, :, C*i:C*(i+1)] = np.array(np.bitwise_and(array, 2**i) > 0, dtype=dtype)
    return bits


def bitsToarray(bits, n=8, dtype=np.uint8):
    H, W, N = bits.shape
    C = N / n
    bits = np.array(bits, dtype=dtype)
    array = np.zeros((H, W), dtype=dtype)
    for i in range(n):
        array += (2**i)*bits[:, :, i]
    return array
