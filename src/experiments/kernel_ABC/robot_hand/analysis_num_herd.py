import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 28})
from scipy.stats import sem
import matplotlib._color_data as mcd
from matplotlib.lines import Line2D



methods = ["SPD", "SPD_chol", "SPD_euclid"]
labels = ["Riemannian", "Cholesky", "Euclidean"]
color_names = ['orange','maize','steel blue']
colors = [mcd.XKCD_COLORS["xkcd:{}".format(x)] for x in color_names]
num_herds = np.array([5, 10, 15, 20]).astype(int)
custom_legend_lines = [Line2D([0], [0], color=colors[0], lw=10),
                Line2D([0], [0], color=colors[1], lw=10),
                Line2D([0], [0], color=colors[2], lw=10)]


sim_results = []
sim_sems = []
for num_herd in num_herds:
    folder = "results/saved/numherd{}".format(num_herd)
    test_results = [np.load("{}/test_results_{}.npy".format(folder, method)) for method in methods]
    sim_err = [test_results[idx][0, :, -1].mean() for idx in np.arange(len(test_results))]
    sim_sem = [sem(test_results[idx][0, :, -1]) for idx in np.arange(len(test_results))]
    sim_results.append(sim_err)
    sim_sems.append(sim_sem)
sim_results = np.stack(sim_results)
sim_sems = np.stack(sim_sems)

fig, axs = plt.subplots(1, figsize=(10,9))
[axs.plot(num_herds, sim_results[:, idx],
          '-o', color=colors[idx], markersize=10,
          label=labels[idx]) for idx in np.arange(len(methods))]
[axs.fill_between(num_herds,
                 sim_results[:, idx] + sim_sems[:, idx],
                 sim_results[:, idx] - sim_sems[:, idx],
                color=colors[idx],
                  alpha=0.4) for idx in np.arange(len(methods))]
axs.set_xticks(num_herds)

ax2 = axs.twiny()
ax2.set_xticks( axs.get_xticks() )
ax2.set_xbound(axs.get_xbound())
ax2.set_xlabel("Number of Simulations")
ax2.set_xticklabels([int(x * 20) for x in axs.get_xticks()])
ax2.xaxis.labelpad = 10
# ax2.legend(prop={'size': 22})


axs.set_xlabel("Number of herded samples")
axs.set_ylabel("Simulation Errors")
axs.legend(custom_legend_lines, labels,
           loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.4),
          fancybox=False, ncol=3, prop={'size': 24}, markerscale=2)
           # columnspacing=0.4, handletextpad=0.1)
plt.tight_layout()
plt.savefig('results/all_methods_over_nherds.svg')
plt.show()
plt.close()
