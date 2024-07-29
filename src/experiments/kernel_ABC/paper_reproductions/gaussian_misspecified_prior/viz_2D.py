import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def plot_2D_gauss(base_mu, base_sigma, ref_mu=None, ref_sigma=None):
    pass


if __name__ == "__main__":
    # Load true data
    true_mean = np.load('data/true_theta.npy')
    true_Sigma = np.load('data/true_Sigma.npy')
    true_dist = multivariate_normal(mean=true_mean, cov=true_Sigma)

    # Load learned data
    pred_means = np.load('results/theta_preds.npy')

    xlims = (-40,40)
    ylims = (0,80)
    spacing = 0.1
    x, y = np.mgrid[xlims[0]:xlims[1]:spacing, ylims[0]:ylims[1]:spacing]
    pos = np.dstack((x, y))

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    contf = ax.contourf(x, y, true_dist.pdf(pos))
    plt.colorbar(contf)
    ax.set_xlim((-20,20))
    ax.set_ylim((0,80))

    ax.scatter(pred_means[:, 0], pred_means[:, 1], color='red')


    for idx in np.arange(1, pred_means.shape[0]):
        ax.annotate('', xy = (pred_means[idx,0], pred_means[idx,1]),
                    xytext = (pred_means[idx-1,0], pred_means[idx-1,1]),
                    arrowprops=dict(facecolor='black',
                                    arrowstyle='->'))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    plt.close()

