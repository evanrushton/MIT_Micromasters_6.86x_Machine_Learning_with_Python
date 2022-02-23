import numpy as np
import sys

def randomization(n):
    A = np.random.random([n,1])
    return A
print(sys.argv[1])
print(randomization(int(sys.argv[1])))
