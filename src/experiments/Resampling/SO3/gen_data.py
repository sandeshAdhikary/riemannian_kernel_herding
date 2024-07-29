import numpy as np
import torch
from tqdm import trange
from scipy.stats import special_ortho_group
from src.manifold_utils.so3_utils import isSO3
torch.set_default_dtype(torch.double)

if __name__ == "__main__":
    dims = 3
    num_data = 1500
    num_seeds = 20
    for seed in trange(num_seeds, desc="Generating data"):
        np.random.seed(seed)
        data = special_ortho_group.rvs(dims, size=num_data)
        assert isSO3(data), "Data does not satisfy SO3 constraints"
        weights = np.random.normal(size=num_data)
        weights[weights < 0] = weights[weights < 0] * -1
        weights = weights / weights.sum()  # Normalize weights
        np.save('data/weighted_samples_dims{}_dataseed{}.npy'.format(dims, seed),
                dict({
                    'samples': data,
                    'weights': weights
                }))