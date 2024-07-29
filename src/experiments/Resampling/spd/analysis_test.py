import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

methods = ['KernelHerding','KernelHerdingChol','KernelHerdingEuclid', 'OptimalTransport']

fig,axs = plt.subplots(1)
# method = methods[0]
for method in methods:
    sampling_errs = np.load('results/resampling_exp_errs_{}.npy'.format(method))
    mean_errs = np.mean(sampling_errs, axis=0)
    sem_errs = sem(sampling_errs, axis=0)
    axs.plot(np.arange(len(mean_errs)), mean_errs, label=method)
    axs.fill_between(np.arange(len(mean_errs)), mean_errs+sem_errs, mean_errs-sem_errs, alpha=0.2)
    axs.legend()

plt.show()