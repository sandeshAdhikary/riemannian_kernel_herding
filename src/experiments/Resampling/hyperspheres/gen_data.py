import numpy as np
from src.distributions.von_mises import VonMisesMixture3D
import torch
from tqdm import trange

torch.set_default_dtype(torch.double)

def gen_dist(dims, seed=123):
    '''
    Generate samples from a vonMisesMixture distribution with as many components as there are dims
    We'll set the dims poles as the means of the mixture components
    '''

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set up the distribution and generate data
    dist_type = 'VonMisesMixture'
    conc = 1.0
    dist_name = 'sphere_dist_dims{}_seed{}'.format(dims, seed)

    # Generate iid samples from Von Mises mixture distribution
    num_comps = dims
    locs = np.zeros((num_comps, dims))
    for idx in np.arange(num_comps):
        loc = np.zeros(dims)
        loc[idx] = 1.0
        locs[idx, :] = loc

    locs = locs.reshape(num_comps, 1, dims)
    locs = locs / np.linalg.norm(locs, axis=(1, 2))[:, np.newaxis, np.newaxis] # normalize
    concs = np.array([conc]*num_comps)
    weights = np.array([1./num_comps]*num_comps)
    dist = VonMisesMixture3D(locs, concs, weights)
    dist_dict = {'type': dist_type,
                 'locs': locs,
                 'concs': concs,
                 'weights': weights,
                 'seed': seed,
                 'dims': dims
                 }
    np.save('data/{}_dict.npy'.format(dist_name), dist_dict)

    return dist, dist_name

def gen_data_from_dist(dist, dist_name, num_data, seed):
    data = dist.sample(num_data).numpy().squeeze()
    weights = np.random.normal(size=num_data)
    weights[weights < 0] = weights[weights < 0] * -1 #Set weights to be non-negative
    weights = weights/weights.sum() # Normalize weights
    np.save('data/weighted_samples_{}_dataseed{}.npy'.format(dist_name,seed),
            dict({
                'samples': data,
                'weights': weights
            }))

if __name__ == "__main__":
    dims = 4
    dist, dist_name = gen_dist(dims, seed=123)

    num_data = 1500
    print("Num data points:{}".format(num_data))
    num_seeds = 11
    for seed in trange(num_seeds, desc="Generating data"):
        gen_data_from_dist(dist, dist_name, num_data, seed)