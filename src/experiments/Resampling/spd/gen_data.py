import numpy as np
from src.distributions.wishart import InverseWishart
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
    dist_type = 'InverseWishart'
    dist_name = 'spd_dist_dims{}_seed{}'.format(dims, seed)

    # Generate iid samples from the Inverse Wishart distribution
    nu = dims+2 # must be larger than dims
    K = 1.0*torch.diag(torch.rand(dims))
    dist = InverseWishart(nu, K)


    dist_dict = {'type': dist_type,
                 'nu': nu,
                 'K': K,
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
    dims = 3
    dist, dist_name = gen_dist(dims, seed=123)

    num_data = 1000
    num_seeds = 41
    for seed in trange(num_seeds, desc="Generating data"):
        gen_data_from_dist(dist, dist_name, num_data, seed)