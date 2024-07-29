import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
from src.divergences.wasserstein import wasserstein_dist
from src.manifold_utils.euclidean_utils import geodesic as euclid_dist
torch.set_default_dtype(torch.double)

NUM_SEEDS = 5
# NUM_SAMPLES = 100

### Generate some random samples from true_theta, and get distances from y_test

# Load data and form the true distribution
data = np.load('data/covar_est_data_dims3.npy', allow_pickle=True).item()
true_loc = data['true_loc']
true_theta = data['true_theta']
dist = MultivariateNormal(true_loc, true_theta)
y_test = data['y_test']
# num_samples = data['num_test']

# Generate random samples from the true dist
np.random.seed(1234)
torch.manual_seed(1234)
errs = []
for idx in np.arange(NUM_SEEDS):
    samples = dist.sample((y_test.shape[1],)).detach().reshape(y_test.shape[1], -1)
    wass_dist = wasserstein_dist(samples, y_test[idx,:], dist_func=euclid_dist)
    errs.append(wass_dist)

np.save('results/iid-samples.npy', errs)


# Generate some completely random samples for reference
rando_samples = torch.rand((NUM_SEEDS, y_test.shape[1], y_test.shape[2]))
errs = []
for idx in np.arange(NUM_SEEDS):
    wass_dist = wasserstein_dist(rando_samples[idx, :], y_test[idx, :], dist_func=euclid_dist)
    errs.append(wass_dist)

np.save('results/random-samples.npy', errs)

print("Done")



