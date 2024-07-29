import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import ticker

def sph2cart(angles, epsilon=1e-10):
    '''
    Convert 'angles' in spherical cordinate to cartesian coordinates
    Using the convention in: Xu et al (2020)
    Currently, only works for S^{2} (i.e. the "3-dim" sphere})
    input: (num_samples x 2)
            angles[:, 0] = theta \in (0,pi]
            angles[:,1] = phi \in (0,2pi]
    '''

    if len(angles.shape) == 1:
        angles = angles.reshape(1, -1)

    assert angles.shape[1] == 2, "Conversion only implemented for the 3D spheres"

    if torch.is_tensor(angles):
        cos, sin = torch.cos, torch.sin
        stack = torch.stack
        zeros_like, ones_like = torch.zeros_like, torch.ones_like
    else:
        cos, sin = np.cos, np.sin
        stack = np.stack
        zeros_like, ones_like =np.zeros_like, np.ones_like


    # Adjust for possibly negative theta
    neg_thetas = angles[:, 0] < 0
    if any(neg_thetas) == 1:
        # Add pi the phi with negatives thetas
        angles = angles + stack([zeros_like(angles[:, 0]), neg_thetas * np.pi]).T
        # Reverse sign of negative thetas
        angles = angles * stack([-1*neg_thetas + (angles[:, 0] > 0), ones_like(angles[:,0])]).T


    sin_theta_1 = sin(angles[:, 0])
    x = sin_theta_1 * cos(angles[:, 1])
    y = sin_theta_1 * sin(angles[:, 1])

    z = cos(angles[:, 0])

    return stack([x, y, z]).T

def cart2sph(X):
    '''
    Convert X in cartesian coordinats to spherical coordinates
    Currently, only works for X.shape = (num_samples x 3)
    input: (num_samples x 3).
           X[:,0], X[:,1], X[:,2] = x,y,z co-ordinates. Must be normalized across the three dimensions
    output: (num_samples x 2)
            output[:,0], output[:,1] = [\theta \in (0, pi], \phi \in (0, 2pi]]
    '''


    if torch.is_tensor(X):
        theta_1 = torch.atan2(torch.sqrt(X[:, 0]**2 + X[:, 1]**2), X[:, 2])
        assert not torch.any(theta_1 < 0), "theta_1 is negative"

        theta_2 = torch.atan2(X[:, 1], X[:, 0])
        theta_2[theta_2 < 0] += 2 * np.pi # From [-pi,pi] to [0,2pi]

        return torch.stack([theta_1, theta_2]).T
    else:
        theta_1 = np.arctan2(np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2), X[:, 2])
        assert not np.any(theta_1 < 0)

        theta_2 = np.arctan2(X[:, 1], X[:, 0])
        theta_2[theta_2 < 0] += 2 * np.pi # From [-pi,pi] to [0,2pi]

        return np.stack([theta_1, theta_2]).T

def plot_contour_over_angles(axis,angles,values, num_grid_samples):
    CS = axis.contourf(angles[:, 0].reshape(num_grid_samples, num_grid_samples),
                     angles[:, 1].reshape(num_grid_samples, num_grid_samples),
                     values.reshape(num_grid_samples, num_grid_samples))

    plt.colorbar(CS)
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'
    ))
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/4))
    axis.set_xlabel('Theta (Inclination)')
    axis.set_xlim((0, np.pi))

    axis.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'
    ))
    axis.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
    axis.set_ylabel('Phi (Azimuth)')
    axis.set_ylim(0, 2*np.pi)

def plot_probs(log_prob_func,grad_log_prob_func=None):
    '''
    Plot log_prob over the circle
    '''

    num_grid_samples = 100
    theta_1 = np.linspace(0., np.pi, num_grid_samples)
    theta_2 = np.linspace(0., 2*np.pi, num_grid_samples)

    angles = np.array([np.array([a,b]) for a in theta_1 for b in theta_2])

    xyz_points = sph2cart(angles)
    log_probs = log_prob_func(torch.from_numpy(xyz_points))
    #
    fig = plt.figure(figsize=(10, 10))
    if grad_log_prob_func is None:
        num_rows, num_cols = 1, 1
    else:
        num_rows, num_cols = 1, 2
    k = 0

    # # Plot the distribution (in angles)
    k += 1
    contf_ax = fig.add_subplot(num_rows, num_cols, k)
    plot_contour_over_angles(contf_ax, angles, log_probs, num_grid_samples)

    if grad_log_prob_func is not None:
        k += 1
        contf_ax = fig.add_subplot(num_rows, num_cols, k)
        grad_log_probs = grad_log_prob_func(torch.from_numpy(xyz_points))
        grad_log_probs = torch.norm(grad_log_probs, dim=1)
        grad_log_probs = grad_log_probs/max(grad_log_probs)
        plot_contour_over_angles(contf_ax, angles, grad_log_probs, num_grid_samples)

    plt.show()
