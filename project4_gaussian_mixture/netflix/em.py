"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
import numpy.ma as ma
from scipy.special import logsumexp
from common import GaussianMixture

# Calulcate the probability density for data points x in a normal distribution N(mean, var)
def norm_pdf(X, mu, var):
    shape = X.shape
    _, d = shape

    return 1/((2*np.pi*var)**(d/2)) * np.exp(-np.divide(np.linalg.norm(X - mu, axis=1)**2,2*var))

def log_norm_pdf(X, mu, var):
    shape = X.shape
    _, d = shape

    return np.log(2*np.pi*var)*(-d/2) + -np.divide(np.linalg.norm(X - mu, axis=1)**2,2*var)

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    K, d = mixture.mu.shape
    post = np.float64(np.zeros((n, K)))
    log_density = np.float64(np.zeros((n,K)))
    for j in range(K):
        for i in range(n):
            C_u = np.nonzero(X[i, :]) # observed col idx
            card = len(C_u)
            if card > 0:
                log_density[i,j] = np.log(mixture.p[j]+10**-16)+log_norm_pdf(X[i,C_u],mixture.mu[j][C_u],mixture.var[j]*np.identity(card))

    LLs = logsumexp(log_density, axis=1)
    for j in range(K):
        post[:,j] = np.exp(log_density[:,j] - LLs) # divide by rowsum to get multinomial 
    LL = np.sum(LLs)

    return post, LL



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    p = np.zeros((K,))
    mu = np.zeros((K, d))
    var = np.zeros((K,))

    for j in range(K):
        n_hat = np.sum(post[:,j])
        p[j] = n_hat / n
        # Mu Loop
        for m in range(d):
            C_d = np.nonzero(X[:,m])[0]
           # Create boolean mask
            mask = np.zeros(X[:,m].shape,dtype=bool)
            mask[C_d] = True
            n_hat_mu = np.sum(post[:,j][mask])
            if n_hat_mu > 1:
                mu[j,m] = np.sum(np.multiply(post[:,j][mask], X[:,m][mask])) / n_hat_mu
            else: mu[j,m] = mixture.mu[j,m]
        # Variance Loop
        n_hat_var = 0
        var_tmp = np.zeros((K,))
        for i in range(n):
            C_u = np.nonzero(X[i,:])[0] # observed col idx
            card = len(C_u)
           # Create boolean mask
            mask = np.zeros(X[i,:].shape,dtype=bool)
            mask[C_u] = True
            var_tmp[j] += post[i,j] * np.linalg.norm(X[i,:][mask] - mu[j][mask])**2
            n_hat_var += post[i,j]*card
        new_var = var_tmp[j]/n_hat_var
        if  new_var > min_variance:
            var[j] = new_var # update var
        else: var[j] = min_variance
    
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
        mixture = mstep(X, post, mixture)
        iters+=1
        #print(iters, mixture, cost, cost-prev_cost)

    return mixture, post, cost


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, _ = X.shape
    K, d = mixture.mu.shape
    post, LL = estep(X, mixture)
    a = X.copy()
    for i in range(n):
        C_u = np.nonzero(X[i,:])[0] # observed col idx
       # Create boolean mask
        mask = np.ones(X[i,:].shape,dtype=bool)
        mask[C_u] = False 
       # k = np.argmax(post[i,:] + 1)
        a[i,:][mask] = np.sum(np.multiply(np.tile(post[i,:].reshape(-1,1), (1,d)), mixture.mu), axis=0)[mask]
        #print(k, post[i,:], mix.mu, a)
    return a
