import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
import matplotlib._color_data as mcd
from matplotlib.lines import Line2D

filenames = ['iid-samples',
             'random-samples',
             'krabc-results-kernel-herding',
              'krabc-results-kernel-herding-cholesky',
             'krabc-results-kernel-herding-euclid'
             ]
labels = ["True Dist.",
          "Random",
          "Riemannian",
          "Cholesky",
          "Euclidean"]
color_names = ['light peach','light grey', 'orange','maize','steel blue']
colors = [mcd.XKCD_COLORS["xkcd:{}".format(x)] for x in color_names]
custom_legend_lines = [Line2D([0], [0], color=colors[0], lw=10),
                       Line2D([0], [0], color=colors[1], lw=10),
                       Line2D([0], [0], color=colors[2], lw=10),
                       Line2D([0], [0], color=colors[3], lw=10),
                       Line2D([0], [0], color=colors[4], lw=10)]

res_list = [np.load('results/{}.npy'.format(filename), allow_pickle=True) for filename in filenames]
means = [np.mean(res) for res in res_list]
stds = [np.std(res) for res in res_list]

fig, ax = plt.subplots(1)
barlist = ax.bar(labels, means, alpha=0.8)
barlist.xticks = [labels[0], labels[1], " ", " ", " "]
[bar.set_color(color) for (bar,color) in zip(barlist, colors)]
errax = ax.errorbar(labels, means,
             yerr=stds, color='black', fmt='none')
ax.get_xaxis().set_visible(False)
ax.set_xticklabels = [labels[0], labels[1], " ", " ", " "]
plt.ylabel("Simulation Errors")
plt.legend(custom_legend_lines, labels, frameon=False, labelspacing=0.2)
plt.savefig('results/all_models.svg')
plt.show()
print("Done")
