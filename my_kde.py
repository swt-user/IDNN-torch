import torch
import math

def get_dist(x):
    #compute KL-distance matrix for a set of vectors specifie by the matrix x
    x1 = torch.unsqueeze(torch.sum(x*x, dim=1), 1)
    dists = x1 + torch.t(x1) - 2*x.mm(x.t())
    return dists

def get_shape(x):
    dims = x.size()[1]
    N = x.size()[0]
    return dims, N

def logsumexp(x, axis):
    #avoid overflow
    a = x.max()
    return a + torch.log(torch.sum(torch.exp((x - a) * 1.0), axis))

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    dims, N = get_shape(x)
    dists = get_dist(x) * 1.0 / (2*var)
    return (-torch.mean(logsumexp(-dists, axis=1) * 1.0) + math.log(N) + (math.log(2 * math.pi * var) + 1) * dims / 2).item()

def entropy_estimator_bd(x, var):
    # BD-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x, 4*var)
    return val + math.log(0.25) * dims / 2

def kde_condentropy(x, var):
    # Return entropy of a multivariate Gaussian in nats
    dims = x.size()[1]
    return (dims / 2.0) * (math.log(2 * math.pi * var) + 1)

def MI_estimator(x, noise_var):
    """estimate mutual information between X and Y = f(X) + Noise(0, noise_var) using kde mrthod
    input x(torch.tensor), output MI upper bound and lower bound(not tensor)
    noise_var generally takes 0.05(as reference demo) 
    """
    H_Y_given_X = kde_condentropy(x, noise_var)
    H_Y_upper = entropy_estimator_kl(x, noise_var)
    H_Y_lower = entropy_estimator_bd(x, noise_var)
    return H_Y_upper - H_Y_given_X, H_Y_lower - H_Y_given_X