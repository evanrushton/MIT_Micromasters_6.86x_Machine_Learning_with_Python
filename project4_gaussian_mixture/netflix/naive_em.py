"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

# Calulcate the probability density for data points x in a normal distribution N(mean, var)
def norm_pdf(X, mu, var):
    shape = X.shape
    _, d = shape

    return 1/((2*np.pi*var)**(d/2)) * np.exp(-np.divide(np.linalg.norm(X - mu, axis=1)**2,2*var))
    #return 1/(np.sqrt(2*np.pi*var)) * np.exp(-np.divide(np.linalg.norm(X - mu, axis=1)**2,2*var))

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, _ = X.shape
    K, d = mixture.mu.shape
    post = np.float64(np.zeros((n, K)))
    marginal_dist = np.float64(np.zeros((n,K)))
    for j in range(K):
        marginal_dist[:,j] = mixture.p[j]*norm_pdf(X,mixture.mu[j],mixture.var[j])
    L = np.sum(marginal_dist, axis=1)
    for j in range(K):
       post[:,j]=marginal_dist[:,j] / L # divide by rowsum to get multinomial 
    LL = np.sum(np.log(L))

    return post, LL


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
        for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    n_hat = np.sum(post, axis=0) # ( , K)
    p = n_hat / n
    mu =  np.multiply(np.atleast_2d(1/n_hat).T, post.T @ X)
    var = np.sum(np.multiply(np.multiply(1/(n_hat*d), post), (np.linalg.norm(X-mu[:, np.newaxis], axis=2)**2).T), axis=0)
    
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
        for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
        for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = 0 
    cost = 0
    iters=0
    epsilon = 1e-6
    while (prev_cost == 0 or cost - prev_cost >= epsilon*np.abs(cost)):
    #while (prev_cost == 0 or np.abs(cost - prev_cost) >= epsilon):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post)
        iters+=1
        #print(iters, mixture, cost, cost-prev_cost)

    return mixture, post, cost
