import numpy as np
from src.kernels.kernels import Laplacian
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import special_ortho_group
from src.manifold_utils.so3_utils import isSO3 as onManifold


np.random.seed(1)

# Set up the distribution
manifold_type = "SO3"
dims = 3
num_samples = 100
num_trials = 10
all_res = dict()

bandwidths = np.linspace(1e-5, 1, 50)

for bandwidth in tqdm(bandwidths):
    min_eigs = []
    pd_prop = 0
    dists_to_eye = []
    dists_to_ones = []
    for trial in np.arange(num_trials):
        # Generate some random data
        data_x = special_ortho_group.rvs(dims, size=num_samples)

        assert onManifold(data_x), "Data is not Orthogonal"
        assert data_x.dtype == np.double, "Data must be double precision"

        # Set up the kernel
        kernel = Laplacian(bandwidth=bandwidth, manifold=manifold_type)
        K = kernel.kernel_eval(data_x, data_x)

        dist_to_eye = np.linalg.norm(K - np.eye(K.shape[0]))
        dist_to_ones = np.linalg.norm(K - np.ones(K.shape))
        dists_to_eye.append(dist_to_eye)
        dists_to_ones.append(dist_to_ones)

        # Make sure K is symmetric
        if np.linalg.norm(K - K.T) > 1e-7:
            raise (Exception("K is not symmetric!"))

        # Compute minimum eigenvalue of K
        min_eig = np.min(np.linalg.eigh(K)[0])
        min_eigs.append(min_eig)

        # # Check for PD: try cholesky decomposition
        try:
            if np.linalg.cond(K) > 1e7:
                np.linalg.cholesky(K + (1e-7) * np.eye(K.shape[0]))
            else:
                np.linalg.cholesky(K)
            # noise = (1e-8)*np.eye(K.shape[0])
            # np.linalg.cholesky(K + noise)
            pd_prop += 1
        except np.linalg.LinAlgError:
            pass
            # print(np.linalg.cond(K))
            # pass

    pd_prop = 1.0 * pd_prop / num_trials

    mean_dist_to_eye = np.mean(dists_to_eye)
    mean_dist_to_ones = np.mean(dists_to_ones)

    all_res[bandwidth] = [min_eigs, pd_prop, mean_dist_to_eye, mean_dist_to_ones]

min_eigs_avg = np.array([np.mean(all_res[x][0]) for x in all_res.keys()])
min_eigs_max = np.array([np.max(all_res[x][0]) for x in all_res.keys()])
min_eigs_min = np.array([np.min(all_res[x][0]) for x in all_res.keys()])
pd_props = np.array([all_res[x][1] for x in all_res.keys()])
dists_to_eye = np.array([np.mean(all_res[x][2]) for x in all_res.keys()])
dists_to_ones = np.array([np.mean(all_res[x][3]) for x in all_res.keys()])

fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].plot(list(all_res.keys()), pd_props, '-o')
axs[0].set_ylabel('Proportion of PD kernel matrices')
axs[1].set_xlabel('Bandwidth')

axs[1].plot(list(all_res.keys()), np.round(min_eigs_avg, 3), '-o')
axs[1].fill_between(all_res.keys(),
                    min_eigs_min, min_eigs_max,
                    alpha=0.1)

# axs[1].set_ylim((0, 0.5))
axs[1].set_ylabel('Minimum Eigenvalue')
axs[1].set_xlabel('Bandwidth')

axs[2].plot(list(all_res.keys()), dists_to_eye)
axs[2].set_ylabel("Distance to Identity Matrix")
axs[2].set_xlabel('Bandwidth')

axs[3].plot(list(all_res.keys()), dists_to_ones)
axs[3].set_ylabel("Distance to Ones Matrix")
axs[3].set_xlabel('Bandwidth')

plt.suptitle('Distribution: {}'.format(manifold_type))
plt.tight_layout()
plt.savefig('kernel_pd_test/kernel_pd_test_{}'.format(manifold_type))

