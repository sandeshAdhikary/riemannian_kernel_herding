import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

fig, axs = plt.subplots(2)

methods = ["SPD", "SPD_chol","SPD_euclid"]
# methods = ["SPD_chol"]
for method in methods:
    res = np.load('results/test_results_{}.npy'.format(method))
    sim_errs = res[0, :]
    sim_errs_mean = np.mean(sim_errs, axis=0)
    sim_errs_std = np.std(sim_errs, axis=0)

    est_errs = res[1, :]
    est_errs_mean = np.mean(est_errs, axis=0)
    est_errs_std = np.std(est_errs, axis=0)


    axs[0].plot(np.arange(len(est_errs_mean)), est_errs_mean, '-o', label=method)
    axs[0].fill_between(np.arange(len(est_errs_mean)),
                      est_errs_mean+est_errs_std,
                      est_errs_mean-est_errs_std,
                        alpha=0.1)

    axs[1].plot(np.arange(len(sim_errs_mean)), sim_errs_mean, '-x', label=method)
    axs[1].fill_between(np.arange(len(sim_errs_mean)),
                      sim_errs_mean+sim_errs_std,
                      sim_errs_mean-sim_errs_std,
                        alpha=0.1)
axs[0].set_ylabel("Estimation Errors")
axs[1].set_ylabel("Simulation Errors")
axs[0].legend()
axs[1].legend()
plt.savefig('results/test_results_all_methods.png')
