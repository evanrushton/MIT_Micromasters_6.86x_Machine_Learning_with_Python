import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 12
n, d = X.shape
seed = 1

mix, post = common.init(X, K, seed) # random initialization
mix_em, post_em, cost_em = em.run(X, mix, post) # run EM 
X_pred = em.fill_matrix(X, mix_em) 

print(common.rmse(X_gold, X_pred))
