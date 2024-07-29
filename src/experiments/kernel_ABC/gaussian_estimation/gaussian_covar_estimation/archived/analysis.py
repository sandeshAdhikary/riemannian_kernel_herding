import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

exp_name = ['Full SPD Matrix', 'Cholesky Parameterization']
res_files = ['results/gaussian_covar_estimation_results.npy',
             'results/gaussian_covar_estimation_cholesky_results.npy']

fig= plt.figure()
for idx, exp in enumerate(exp_name):
    res = np.load(res_files[idx], allow_pickle=True)
    mean_predictions = np.mean(res, axis=0)
    # std_devs = np.std(res, axis=0)
    sems = sem(res, axis=0)
    plt.plot(np.arange(mean_predictions.shape[0]), mean_predictions, label=exp)
    plt.fill_between(np.arange(mean_predictions.shape[0]),
                     mean_predictions - sems, mean_predictions + sems, alpha=0.3)

plt.legend()
fig.savefig('results/analysis.png')
# plt.show()