import numpy as np
from numpy.random import multivariate_normal as y_cond_dist
from numpy.random import uniform as prior
from src.kr_abc.kr_abc import KernelRecursiveABC
from pymanopt.manifolds import Euclidean
import matplotlib.pyplot as plt
from src.manifold_utils.euclidean_utils import geodesic as euclidean_dist

np.random.seed(123456)

# Set up params of the target distribution
n_dim = 20
# mu = np.zeros(n_dim)
# mu = 1.0*np.array([10,50])
mu = 1.0*np.array([(10,50,90,130,180,280,390,430,520,630,1010,1050,1090,1130,1180,1280,1390,1430,1520,1630)]).reshape(-1)
# mu = np.random.random(n_dim)*(10-1.)
# Sigma = np.eye(n_dim)*40.0
Sigma = np.eye(n_dim)*40
true_theta = mu # The theta we're trying to learn
np.save('data/true_theta.npy', true_theta)
np.save('data/true_Sigma.npy', Sigma)

# Generate training observations
num_obs = 500
y = y_cond_dist(mu, Sigma, size=num_obs)

# Split into training and validation sets
train_prop = 0.75
num_train = int(y.shape[0]*train_prop)
num_valid = y.shape[0] - num_train
all_idx = np.random.permutation(np.arange(y.shape[0]))
y_train, y_valid = y[all_idx[:num_train], :], y[all_idx[num_train:], :]

np.save('data/y_train.npy', y_train)
np.save('data/y_valid.npy', y_valid)