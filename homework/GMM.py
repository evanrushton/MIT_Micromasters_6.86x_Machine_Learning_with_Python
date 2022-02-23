import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def normal_dist(x, mean, var):
  prob_density = 1/np.sqrt(2*np.pi*var) * np.exp(-np.divide((np.subtract(x,mean))**2,2*var))
  return prob_density

xs = [0.2, -0.9, -1, 1.2, 1.8]
posterior = []

for i,x in enumerate(xs):
  num = 0.5*norm.pdf(x, loc=-3, scale=2)
  denom = 0.5*norm.pdf(x, loc=-3, scale=2) + 0.5*norm.pdf(x, loc=2, scale=2)
  num2 = 0.5*normal_dist(x, -3, 4)
  denom2 = 0.5*normal_dist(x, -3, 4) + 0.5*normal_dist(x, 2, 4)
  posterior.append(num2/denom2)
  #print(i, x, num/denom, 'scipy')
  #print(i, x, num2/denom2, 'my function') 

n = len(xs)
x = np.array(xs)
post = np.array(posterior)
n_hat = np.sum(post)
mu_hat = np.sum(x@post)/n_hat
p_hat = 1/n * n_hat
s_hat = np.sum(post@(x-mu_hat)**2)/n_hat

print(p_hat, mu_hat, s_hat)


