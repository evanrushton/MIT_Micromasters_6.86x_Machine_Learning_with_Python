import numpy as np
import sys

def operations(h,w):
    A = np.random.random ([h,w])
    B = np.random.random ([h,w])
    return A, B, A+B

print(operations(int(sys.argv[1]), int(sys.argv[2])))
