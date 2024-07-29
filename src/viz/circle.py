import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import ticker

def cart2sph(x):
    '''
    Convert x = [cos\theta, sin\theta] into an angle within [0,2pi]
    '''

    if torch.is_tensor(x):
        return torch.remainder(torch.atan2(x[:, 1], x[:, 0]), 2*np.pi)
    else:
        return np.mod(np.arctan2(x[:, 1], x[:, 0]), 2*np.pi)

def sph2cart(thetas):
    '''
    Convert angles into x = [cos\theta, sin\theta]
    '''
    return np.stack([np.cos(thetas), np.sin(thetas)]).T

def plot_probs(log_prob_func, grad_log_prob_func=None):
    '''
    Plot log_prob over the circle
    '''

    if not grad_log_prob_func:
        print ("Grad log prob viz not implemented. Just plotting log_probs")

    angles = np.linspace(0, 2*np.pi, 100)
    x = sph2cart(angles)
    log_probs = log_prob_func(torch.from_numpy(x))

    fig = plt.figure(figsize=(10, 10))
    num_rows, num_cols = 1, 1
    k = 0

    # Plot the distribution (in angles)
    k += 1
    pdf_ax = fig.add_subplot(num_rows, num_cols, k)
    pdf_ax.plot(angles,log_probs.reshape(-1),'-o')
    # pdf_ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    #     lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'
    # ))
    # pdf_ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/4))
    # pdf_ax.set_xlabel('Theta')
    # pdf_ax.set_xlim((0, 2*np.pi))
    plt.show()
