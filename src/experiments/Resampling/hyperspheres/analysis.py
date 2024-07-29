import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import matplotlib
from matplotlib.lines import Line2D
matplotlib.rcParams.update({'font.size': 16})
import matplotlib._color_data as mcd
color_names = ['orange','steel blue','greenish']
colors = [mcd.XKCD_COLORS["xkcd:{}".format(x)] for x in color_names]
labels = ['Riemannian Herding', 'Euclidean Herding',
          'OT']
custom_legend_lines = [Line2D([0], [0], color=colors[0], lw=10),
                Line2D([0], [0], color=colors[1], lw=10),
                Line2D([0], [0], color=colors[2], lw=10)]
xticks = np.arange(0, 1200, 200)


results = np.load('results/resampling_exp_errs.npy', allow_pickle=True).item()
exp_names = results.keys()
fig, axs = plt.subplots(1, 1)
start_idx = 0 # Sometimes might be cleaner to skip the first few
for idx, exp in enumerate(exp_names):
    errs = np.stack(results[exp])
    means = np.mean(errs, axis=0)
    sems = np.std(errs, axis=0)
    mins = means - sems
    maxs = means + sems
    x = np.arange(start_idx, means.shape[0])*5
    axs.plot(x, means[start_idx:], label="{}".format(labels[idx]),
             color=colors[idx])
    if exp != "herding_errs":
        axs.axhline(means[-1], ls='--', alpha=0.4, color=colors[idx])
    axs.set_xticks(xticks)
    axs.fill_between(x, mins, maxs, alpha=0.1, color=colors[idx])
    axs.set_ylim((0,0.22))
# plt.yscale('log')
plt.ylabel("Sampling Errors")
plt.xlabel("Number of resamples")
plt.legend(custom_legend_lines, labels, frameon=False)
fig.savefig('results/resampling_exp_errs.svg')
plt.show()


# exp_names = ['Kernel Herding', 'Optimal Transport']
# res_file_paths = ['results/kernel_herding_errs.npy', 'results/optimal_transport_errs.npy']
# errs = np.array([np.load(x) for x in res_file_paths ])
# means = np.array([np.mean(x, axis=0) for x in errs]) # Means over seeds
# sems = np.array([sem(x) for x in errs]) # Standard error over seed
# mins = means - sems
# maxs = means + sems
#
# fig, axs = plt.subplots(1, 1)
# for (mean, min, max, name) in zip(means, mins, maxs, exp_names):
#     start_idx = 0  # Sometimes might be cleaner to skip the first few
#     x = np.arange(start_idx, mean.shape[0])
#     axs.plot(x, mean[start_idx:], label="{}".format(name))
#     axs.fill_between(x, mean[start_idx:] - min[start_idx:],
#                      mean[start_idx:] + max[start_idx:], alpha=0.1)
#     axs.set_yscale('log')
# plt.legend()
# fig.savefig('results/resampling_exp_errs.png')
# # plt.show()
# print("Done")