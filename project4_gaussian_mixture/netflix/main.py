import numpy as np
import kmeans
import common
import naive_em
import em
from matplotlib import pyplot as plt

#X = np.loadtxt("toy_data.txt")
X = np.loadtxt("netflix_incomplete.txt")


### Run multiple k-means and EMs and find best initialization seed
K = [1,2,3,4]
k = len(K)
seeds = [0,1,2,3,4]
mixtures_k, posts_k, costs_k, seeds_k, mixtures_em, posts_em, costs_em, seeds_em = k*[[]], k*[[]], k*[10000], k*[0], k*[[]], k*[[]], k*[-10000000], k*[0]
for k in K:
  for seed in seeds:
    mix, post = common.init(X, k, seed) # random initialization
   # mix_k, post_k, cost_k = kmeans.run(X, mix, post) # run kmeans 
   # mix_em, post_em, cost_em = naive_em.run(X, mix, post) # run EM 
    mix_em, post_em, cost_em = em.run(X, mix, post) # run EM 
    # Only keep max values
    print(k, seed, cost_em, costs_em)
   # if cost_k < costs_k[k-1]:
   #     mixtures_k[k-1] = mix_k
   #     posts_k[k-1] = post_k
   #     costs_k[k-1] = cost_k
   #     seeds_k[k-1] = seed
    if cost_em > costs_em[k-1]:
        mixtures_em[k-1] = mix_em
        posts_em[k-1] = post_em
        costs_em[k-1] = cost_em
        seeds_em[k-1] = seed
#  common.plot(X, mixtures_em[k-1], posts_em[k-1],f'EM K={k}, seed={seeds_em[k-1]}')
#  common.plot(X, mixtures_k[k-1], posts_k[k-1],f'K-means K={k}, seed={seeds_k[k-1]}')

### Find Bayesian Information Criterion for values of K
k = len(K)
bic = k*[0]
for k in K:
  bic[k-1] = common.bic(X, mixtures_em[k-1], costs_em[k-1])

plt.plot(K, bic)
plt.show()
print(bic) 


### Run Naive EM
# k=3
# seed=0
# mix, post = common.init(X, k, seed) # random initialization
# mix, post, cost = naive_em.run(X, mix, post)
# print(mix, post.shape)
# common.plot(X, mix, post, f'EM k={k}, seed={seed}')
print(costs_em)
