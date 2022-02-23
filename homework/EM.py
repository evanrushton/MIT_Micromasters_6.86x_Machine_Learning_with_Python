import numpy as np
from matplotlib import pyplot as plt

theta = {'p':[0.5, 0.5], 'mu':[6, 7], 's2':[1, 4]}
thetas = [theta]
x = np.array([-1, 0, 4, 5 ,6], dtype='float64')
iters = 0
thresh = 10**-2
old_like = 0


### Helper Methods ###

# Calulcate the probability density for data points x in a normal distribution N(mean, var)
def normal_dist(x, mean, var):
  prob_density = 1/np.sqrt(2*np.pi*var) * np.exp(-np.divide((np.subtract(x,mean))**2,2*var))
  return prob_density

# Return log-likelihood for a given multinomial
def log_likelihood(x, theta):
  probs = mixture_probs(x, theta)
  #print(probs, np.sum(probs))
  return np.sum(np.log(probs))

# Return mixture probabilities for data points x in two guassian distributions N(mu_1, s2_1) and N(mu_2, s2_2)
def mixture_probs(x, theta):
  return theta['p'][0]*normal_dist(x, theta['mu'][0], theta['s2'][0]) + theta['p'][1]*normal_dist(x, theta['mu'][1], theta['s2'][1])

# Return the posterior probabilities and cluster assignment for data points x with a given set of params theta
def E_step(x, theta):
  n = len(theta['p'])
  m = len(x)
  result = {'assignment':[None] * m, 'probs':np.zeros((n,m))}
  denom = mixture_probs(x, theta)
  for i in range(n):
    num = theta['p'][i]*normal_dist(x, theta['mu'][i], theta['s2'][i])
    result['probs'][i] = num/denom
  result['assignment'] = np.argmax(result['probs'], axis=0)
  return result 

# Return updated theta for data points x given posterior probabilities
def M_step(x, probs):
  k = probs.shape[0] # num rows is number of clusters
  n = len(x)
  result = {'p':[0.5, 0.5], 'mu':[6, 7], 's2':[1, 4]}
  for j in range(k):
    n_hat = np.sum(probs[j])
    result['mu'][j] = 1/n_hat * np.sum(probs[j] @ x)
    result['s2'][j] = 1/n_hat * np.sum(probs[j] @ (x - result['mu'][j])**2)
    result['p'][j] = n_hat/n
  return result 

# plot the data and overlay gaussians
def plot_data(x, thetas, iters, labels):
  fig, ax = plt.subplots(figsize=(9,6))
  #plt.style.use('fivethirtyeight')
  x_vals = np.linspace(-5, 9, 70)
  s = 0.3
  for i in range(iters):
    ax.fill_between(x_vals, normal_dist(x_vals, thetas[i]['mu'][0], thetas[i]['s2'][0]), color='blue', alpha=s*(i+1)/(iters))
    ax.fill_between(x_vals, normal_dist(x_vals, thetas[i]['mu'][1], thetas[i]['s2'][1]), color='red', alpha=s*(i+1)/(iters))
  cdict = {0: 'blue', 1: 'red'}
  for l in np.unique(labels): # plot points based on label
    ix = np.where(labels == l)
    xs = x[ix]
    ys = [0]*len(xs) 
    ax.plot(xs, ys, marker='.', ms=20, c=cdict[l], mec='black', mew=2)
  ax.set_xlim([-5,9])
  ax.set_ylim([-.05, 1])
  ax.set_xlabel('1D values')
  ax.set_ylabel('Probability Density')
  #ax.set_yticklabels([])
  ax.set_title(f'Gaussian Mixture across {iters} EM iterations')
  #plt.savefig('normal_curve.png', dpi=72, bbox_inches='tight')
  plt.show()

### Algorithm ###
new_like = log_likelihood(x, theta)
while(np.abs(old_like - new_like) >= thresh):
  old_like = new_like
  posterior = E_step(x, theta)
  theta = M_step(x, posterior['probs'])
  iters += 1
  thetas.append(theta)
  new_like = log_likelihood(x, theta)

#print(f'iters: {iters}\n posterior: {posterior}\n thetas: {thetas}') 
labels = posterior['assignment']
plot_data(x, thetas, iters, labels)

