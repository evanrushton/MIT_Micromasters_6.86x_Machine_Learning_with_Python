import numpy as np
import kmeans
import common
import naive_em
import em
from matplotlib import pyplot as plt

X = np.loadtxt("netflix_incomplete.txt")
seeds=[0, 1, 2, 3, 4]
k=12
for seed in seeds:
    mix, post = common.init(X, k, seed) # random initialization
    mix_em, post_em, cost_em = em.run(X, mix, post) # run EM 
    print(k, seed, cost_em)
