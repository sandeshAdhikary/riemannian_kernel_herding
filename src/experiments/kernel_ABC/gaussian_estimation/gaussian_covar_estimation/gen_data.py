from src.manifold_utils.spd_utils import num_lower_triu_elems, chol_to_spd, vec_to_chol
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import numpy as np
torch.set_default_dtype(torch.double)

dims = 3
num_train = 1000
num_test = 1000
num_seeds = 10 #Create this many random versions of y_train and y_test

# Generate SPD covariance matrix
chol_vec_dim = num_lower_triu_elems(dims) # Get number of elements in lower triangle
true_theta = torch.rand(chol_vec_dim) # Generate random elements in lower triangle
true_theta = chol_to_spd(vec_to_chol(true_theta, dims=dims)) # Convert lower triangle into full SPD
try:
    # Will fail if true_theta is not SPD
    torch.cholesky(true_theta)
except:
    raise(ValueError("True theta is not SPD"))

# Generate the mean element
true_loc = torch.zeros(dims).reshape(1, -1)

# Generate the distribution
dist = MultivariateNormal(true_loc, true_theta)

# Generate training samples from dist
y_train = []
for idx in np.arange(num_seeds):
    y_train.append(dist.sample((num_train,)).detach().reshape(num_train, -1))
y_train = torch.stack(y_train)

# Generate test samples from dist
y_test = []
for idx in np.arange(num_seeds):
    y_test.append(dist.sample((num_test,)).detach().reshape(num_test, -1))
y_test = torch.stack(y_test)


# Compile everything into a dict and save
data = {
    'true_theta': true_theta,
    'dims': dims,
    'true_loc': true_loc,
    'y_train': y_train,
    'y_test': y_test,
    'num_train': num_train,
    'num_test': num_test,
    'num_seeds': num_seeds
}
np.save('data/covar_est_data_dims{}.npy'.format(dims), data)
