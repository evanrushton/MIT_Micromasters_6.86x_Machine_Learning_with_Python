import numpy as np
import sys

def norm(A, B):
    print(type(A), type(B))
    s = A+B
    return np.linalg.norm(s)
print(norm(np.array(sys.argv[1]),np.array(sys.argv[2]))) 
