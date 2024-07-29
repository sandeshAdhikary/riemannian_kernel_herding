import numpy as np
import torch
from src.distributions.von_mises import VonMises3D
from pymanopt.manifolds import Euclidean, Sphere
from src.kr_abc.kr_abc import KernelRecursiveABC
import matplotlib.pyplot as plt
from src.manifold_utils.sphere_utils import geodesic as sphere_dist
from src.manifold_utils.sphere_utils import project as sphere_project
import torch

np.random.seed(123)
torch.manual_seed(123)

seeds = [0] #, 1, 2, 3, 4]
n_iters = 25
n_herd = 25

# Set up the target distribution and data
# dims = 3
dims = 6
loc = np.zeros(dims).reshape(1, -1)
loc[:, 0] = 1.
loc = loc / np.linalg.norm(loc)  # Normalized so its on the sphere
true_loc = loc
conc = np.array([5.0]).reshape(1, -1)  # Defines concentration around loc
dist = VonMises3D(loc, conc)
theta_dist_func = sphere_dist
y_dist_func = sphere_dist
project_func = sphere_project

for seed in seeds:
    print("Running seed: {}".format(seed))


    def target_simulator(mu_list):
        '''
        Generate samples from the simulator for a list of estimated means mu_list
        :param mu_list:
        :return:
        '''

        y_sampled = np.zeros((mu_list.shape))
        for idx in np.arange(mu_list.shape[0]):
            mu = sphere_project(mu_list[idx, :].reshape(1, -1))
            target_dist = VonMises3D(mu, conc)  # Use the same conc as target
            y_sampled[idx, :] = target_dist.sample(1).reshape(-1).double().numpy()

        return y_sampled

    # Generate data
    num_data = 1000

    y = dist.sample(num_data).numpy().reshape(-1, dims)

    # Set up prior sampler
    prior_loc = np.ones(dims).reshape(1, -1)
    prior_loc = prior_loc / np.linalg.norm(prior_loc)
    prior_conc = conc # Same conc as target distribution
    prior_dist = VonMises3D(prior_loc, conc)

    def prior_sampler(num_samples):
        return prior_dist.sample(num_samples).reshape(num_samples, -1).double().numpy()

    # Set up the true manifold
    true_manifold = Sphere(dims)

    # Set up kernel herding params
    exps = [["Euclidean", "Euclidean"],
            ["Euclidean", "Sphere"],
            ["Sphere", "Euclidean"],
            ["Sphere", "Sphere"]]

    #### Set tuned regs and bandwidths here
    regs = [1e-10, 1e-10,  0.0001, 0.0001]
    bandwidth_scales = [3.29, 0.004, 0.0016, 0.1]

    exps = [exps[3]]
    regs = [regs[3]]
    bandwidth_scales = [bandwidth_scales[3]]

    for exp, reg, bandwidth_scale in zip(exps, regs, bandwidth_scales):
        kernel_theta_type = exp[0]
        opt_manifold_type = exp[1]

        print("Running: {} Kernel | {} Optimization".format(kernel_theta_type, opt_manifold_type))

        # Set up KR-ABC params
        kernel_y_type = "Sphere"


        # Run recursive kernel ABC
        theta_estimates, theta_herds = KernelRecursiveABC(y, n_iters, prior_sampler,
                                                          target_simulator, None,
                                                          kernel_y_type, kernel_theta_type,
                                                          opt_manifold_type, bandwidth_scale,
                                                          n_herd, reg, true_manifold,
                                                          true_theta=true_loc,
                                                          theta_dist_func=theta_dist_func,
                                                          y_dist_func=y_dist_func,
                                                          project_func=project_func)

        # Save theta estimates to file
        np.save('results/errs_{}Kernel_{}Optimization_seed{}.npy'.format(kernel_theta_type,opt_manifold_type, seed),
                theta_herds)

        # errs_over_time = theta_estimates - np.expand_dims(true_loc,0).repeat(n_iters,0)
        errs_over_time = np.linalg.norm(theta_estimates - true_loc, axis=(1))

        plt.plot(errs_over_time, '-o')
        plt.title("{} | {}".format(exp[0], exp[1]))
        plt.show()

        # Compute error of theta_estimates wrt the true theta
        # Compute error for theta_estimates
        # errs = sphere_dist(theta_estimates[-1, :].reshape(1, dims), true_loc)
        # errs = err.reshape(-1)[0])

        # fig, axs = plt.subplots(1,1)
        # axs.plot(errs, '-o')
        # axs.set_yscale('log')
        # axs.set_xlabel("Iterations")
        # axs.set_ylabel("Error")
        # plt.savefig('results/errs_{}Kernel_{}Optimization_seed{}.png'.format(kernel_theta_type, opt_manifold_type,
        #                                                                      seed))
