import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import matplotlib
import matplotlib._color_data as mcd
from matplotlib.lines import Line2D
matplotlib.rcParams.update({'font.size': 16})

methods = ['kernel-herding', 'kernel-herding-char1', 'kernel-herding-euclid', 'optimal-transport-pf', 'optimal-transport-pf-cayley']
labels = ['Riemannian Herding', 'Riemannian Herding (Char)', 'Euclidean Herding', 'OT', 'OT + Cayley Centroid']

file_names = ['resampling_exp_errs_{}.npy'.format(x) for x in methods]
color_names = ['orange', 'lilac', 'steel blue', 'greenish', 'brown'][0:len(methods)]

colors = [mcd.XKCD_COLORS["xkcd:{}".format(x)] for x in color_names]

custom_lengend_lines = [Line2D([0], [0], color=color, lw=10) for color in colors]


fig, axs = plt.subplots(1, 1)
start_idx = 0 # Sometimes might be cleaner to skip the first few
xticks = np.arange(0, 1500, 250)
for idx, file in enumerate(file_names):
    results = np.load('results/{}'.format(file), allow_pickle=True).item()
    errs = np.array(results['errs'])
    exp_name = results['params']['resampling_technique']
    means = np.mean(errs, axis=0)
    sems = np.std(errs, axis=0)
    mins = means - sems
    maxs = means + sems
    x = np.arange(start_idx, means.shape[0])*5
    axs.plot(x, means[start_idx:], label="{}".format(labels[idx]), color=colors[idx])
    axs.set_xticks(xticks)
    axs.fill_between(x, mins, maxs, alpha=0.3, color=colors[idx])
plt.ylabel("Sampling Errors")
plt.xlabel("Number of resamples")
axs.legend(custom_lengend_lines, labels, frameon=False)
fig.savefig('results/resampling_exp_errs.svg')
plt.show()
plt.close()