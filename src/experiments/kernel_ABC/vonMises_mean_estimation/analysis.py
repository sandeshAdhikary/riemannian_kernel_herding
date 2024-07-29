import numpy as np
import matplotlib.pyplot as plt
from src.manifold_utils.sphere_utils import geodesic
from scipy.stats import sem


# Set up true distribution
dims = 6
data_dir = "results"
loc = np.zeros(dims).reshape(1, -1)
loc[:, 0] = 1.
loc = loc / np.linalg.norm(loc)  # Normalized so its on the sphere
true_loc = loc

exps = [["Euclidean", "Euclidean"],
        # ["Euclidean", "Sphere"],
        # ["Sphere", "Euclidean"],
        ["Sphere", "Sphere"]]

all_seeds = [[0, 1, 2, 3, 4],
             # [0, 1, 2, 3, 4],
             # [0, 1, 2, 3, 4],
             [0, 1, 2, 3, 4]]

fig, axs = plt.subplots(1, 1)

for exp, seeds in zip(exps, all_seeds):
    errs_over_seeds = []
    for seed in seeds:
        kernel_type, opt_type = exp[0], exp[1]
        file_name = "{}/errs_{}Kernel_{}Optimization_seed{}.npy".format(data_dir, kernel_type, opt_type, seed)
        herded_samples = np.load(file_name)
        theta_estimates = herded_samples[:, 0, :]  # Use first herded sample as estimate
        errs = geodesic(theta_estimates, true_loc).reshape(-1)
        errs_over_seeds.append(errs)

    errs_over_seeds = np.stack(errs_over_seeds)
    avg_err_over_seeds = np.mean(errs_over_seeds, axis=0)
    std_over_seeds = sem(errs_over_seeds, axis=0)
    min_err_over_seeds = avg_err_over_seeds - std_over_seeds
    max_err_over_seeds = avg_err_over_seeds + std_over_seeds

    start_idx = 0  # Sometimes might be cleaner to skip the first few
    x = np.arange(start_idx, avg_err_over_seeds.shape[0])
    axs.plot(x, avg_err_over_seeds[start_idx:], label="{} Kernel | {} Optimization".format(kernel_type, opt_type))
    axs.fill_between(x, avg_err_over_seeds[start_idx:] - std_over_seeds[start_idx:],
                     avg_err_over_seeds[start_idx:] + std_over_seeds[start_idx:], alpha=0.1)
    # axs.fill_between(x, min_err_over_seeds[start_idx:], max_err_over_seeds[start_idx:], alpha=0.1)

axs.set_yscale('log')
# axs.set_xlabel("Iterations")
# axs.set_ylim((0,0.4))
axs.set_ylabel("Error")
plt.legend()
plt.savefig('results/errs.png')