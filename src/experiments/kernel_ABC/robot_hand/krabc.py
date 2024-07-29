import numpy as np
import geoopt
from src.manifold_utils.product import get_submanifold_tensors
from src.kernels.kernels import Laplacian, ProductKernel
from src.manifold_utils.euclidean_utils import geodesic as euclid_dist_func
from src.manifold_utils.sphere_utils import geodesic as sphere_dist_func
from src.manifold_utils.sphere_utils import project as sphere_project_func
from src.manifold_utils.spd_utils import project as spd_project_func
from src.manifold_utils.spd_utils import geodesic as spd_dist_func
from mj_envs.hand_manipulation_suite.hand_v0 import HandEnvV0 as ENV
import torch
from src.manifold_utils.so3_utils import rotmat_to_quats, quats_to_rotmat
from src.kr_abc.kr_abc_combined import KernelRecursiveABC
import matplotlib.pyplot as plt
from src.experiments.kernel_ABC.robot_hand.gen_data import run_hand_simulation
import copy
from multiprocessing import Pool, cpu_count
import time
import matplotlib.animation as animation
from src.manifold_utils.spd_utils import chol_to_spd,vec_to_chol, num_lower_triu_elems
import warnings
from scipy.stats import sem

torch.set_default_dtype(torch.double)
SCALER = 5 # Apply this scaling to SPD matrices generated by prior sampler
NUM_PROCESSORS = 8

def inertia_estimation_product_manifold(y_train, y_test, true_theta, actions,
                                        params, method, plot=True):
    '''
    Learn the inertia matrix as a product manifold
    If method == "quat", learn as a product manifold of Euclidean + Quaternions
    If method = "quat_euclid", learn as a product manifold of Euclidean + Euclidean (then project the second one)
    '''
    valid_methods = ["quat", "quat_euclid"]
    assert method in valid_methods

    # Unpack variable parameters
    num_herd = params['num_herd']
    theta_bandwidth_diags = params['theta_bandwidth_diags']
    theta_bandwidth_quats = params['theta_bandwidth_quats']
    theta_bandwidths = [theta_bandwidth_diags, theta_bandwidth_quats]
    theta_reg = params['theta_reg']
    lr = params['lr']
    num_epochs = params['num_epochs']
    num_iters = params['num_iters']
    num_processors = params['num_processors']

    # Set fixed parameters
    y_manifold_name = "Euclidean"
    dims = 3 # The system is actually 3 dimensional
    numeric_backend = "pytorch"
    y_bandwidth = None
    adapt_y_bandwidth = True
    adapt_theta_bandwidth = False

    # Get environment
    seed = 1234 # All simulations should start the same way
    env = ENV()
    env.seed(seed)
    env.reset()

    # Set up the manifolds
    if method == "quat":
        theta_manifold_names = ["Euclidean", "Sphere"]
        theta_manifold = geoopt.manifolds.product.ProductManifold(
            (geoopt.Euclidean(), 3),
            (geoopt.Sphere(), 4)
        )
        riemannian_opt = True
    elif method == "quat_euclid":
        theta_manifold_names = ["Euclidean", "Euclidean"]
        theta_manifold = geoopt.manifolds.product.ProductManifold(
            (geoopt.Euclidean(), 3),
            (geoopt.Euclidean(), 4)
        )
        riemannian_opt = False

    # Function to unwrap the consitituent manifolds
    def sub_tensor_fn(x, idx):
        return get_submanifold_tensors(x, idx, theta_manifold)

    # Set up the kernels
    kernel_theta_one = Laplacian(bandwidth=theta_bandwidths[0], manifold=theta_manifold_names[0])
    kernel_theta_two = Laplacian(bandwidth=theta_bandwidths[1], manifold=theta_manifold_names[1])
    kernel_theta = ProductKernel([kernel_theta_one, kernel_theta_two], sub_tensor_fn)
    kernel_y = Laplacian(bandwidth=y_bandwidth, manifold=y_manifold_name)

    # Function to standardize thetas into diag+quaternion form
    def standardize_thetas(theta_list):
        # Make the first 3 entries non-negative
        theta_list_std = theta_list.clone()
        theta_list_std[:, 0:3] = torch.abs(theta_list_std[:, 0:3])
        if method == "quat_euclid":
            # Project "quaternions" onto sphere
            theta_list_std[:, 3:] = sphere_project_func(theta_list_std[:, 3:])
        return theta_list_std

    # Define the simulator function
    def observation_simulator(theta_list):
        # First standardize the thetas
        theta_list_std = standardize_thetas(theta_list)

        if num_processors > 1:
            multi_proc_args = [[copy.deepcopy(env), theta, actions] for theta in theta_list_std]
            with Pool(min(num_processors, cpu_count() - 1)) as pool:  # Use one less than total cpus to prevent freeze
                results = pool.starmap(run_hand_simulation, multi_proc_args)
        else:
            results = []
            for sim_id in np.arange(theta_list_std.shape[0]):
                results.append(run_hand_simulation(env, theta_list_std[sim_id, :], actions))
        results = torch.from_numpy(np.stack(results))
        return results

    # Define the prior sampler for thetas
    def prior_theta_sampler(num_samples, scaler=SCALER):
        torch.manual_seed(1234)  # Reset seed for consistency of initial samples
        ## Generate random SPD matrices
        samples = scaler * torch.rand((num_samples, num_lower_triu_elems(dims)))
        samples = torch.stack([chol_to_spd(vec_to_chol(x, dims=dims)) for x in samples])
        ## Get their eigendecompositions
        D, U = torch.symeig(samples, eigenvectors=True)
        assert torch.all(D > 0), "Eigvalues of SPD matrix is not positive"
        ## Convert U matrices into quaternions
        Q = rotmat_to_quats(U)

        return torch.hstack([D, Q])

    # Define the distance function to compare thetas with true theta
    def theta_dist_func(theta_est, theta_true):
        # Convert theta_list to the standard form
        theta_est_std = standardize_thetas(theta_est)

        # Convert theta_est_std in standard form to full SPD matrices
        theta_est_D = theta_est_std[:, 0:3]
        theta_est_R = quats_to_rotmat(theta_est_std[:,3:])
        theta_est_SPD = np.zeros((theta_est_std.shape[0], dims, dims))
        for idx in np.arange(theta_est_std.shape[0]):
            theta_est_SPD[idx,:,:] = theta_est_R[idx,:,:] @ np.diag(theta_est_D[idx,:]) @ theta_est_R[idx,:,:].T

        # Convert thea_true in standard form to full SPD matrices
        theta_true_D = theta_true[:, 0:3]
        theta_true_R = quats_to_rotmat(theta_true[:,3:])
        theta_true_SPD = np.zeros((theta_true.shape[0], dims, dims))
        for idx in np.arange(theta_true.shape[0]):
            theta_true_SPD[idx,:,:] = theta_true_R[idx,:,:] @ np.diag(theta_true_D[idx,:]) @ theta_true_R[idx,:,:].T

        # Get distances between theta_est and theta_true in terms of SPD geodesic distance
        return spd_dist_func(theta_est_SPD, theta_true_SPD)


        # # Get distances based on diag + quaternion form
        # dist_euclid = euclid_dist_func(theta_est_std[:, 0:3], theta_true[:, 0:3])
        # dist_quat = sphere_dist_func(theta_est_std[:, 3:], theta_true[:, 3:])
        # return dist_euclid + dist_quat

    # Set up KRABC using y_train
    krabc = KernelRecursiveABC(y_train, num_herd, prior_theta_sampler, observation_simulator,
                               theta_manifold, kernel_y, kernel_theta,
                               reg=theta_reg,
                               numeric_backend=numeric_backend,
                               adapt_y_bandwidth=adapt_y_bandwidth,
                               adapt_theta_bandwidth=adapt_theta_bandwidth,
                               riemannian_opt=riemannian_opt,
                               y_dist_func=euclid_dist_func,
                               true_theta=None,
                               theta_dist_func=theta_dist_func,
                               theta_project_func=None,
                               lr=lr,
                               num_epochs=num_epochs)
    theta_estimates, _ , _ = krabc.run_estimator(num_iters=num_iters)

    # Get estimation and simulation errors
    simulation_errs = []
    simulated_samples_list = []
    estimation_errs = []

    for idx in np.arange(theta_estimates.shape[0]):
        # estimation errors
        estimation_err = theta_dist_func(theta_estimates[idx,:].reshape(1,-1), true_theta.reshape(1,-1))
        estimation_errs.append(estimation_err.item())

        # simulation errors
        simulated_samples = observation_simulator(torch.stack([theta_estimates[idx, :]]))
        simulation_err = torch.norm((simulated_samples - y_test).mean(dim=0))
        simulation_errs.append(simulation_err)
        simulated_samples_list.append(simulated_samples)

    standardized_theta_estimates = standardize_thetas(theta_estimates[-1, :].reshape(1,-1))
    print('Final Theta Estimate: {}'.format(standardized_theta_estimates))
    print('True Theta: {}'.format(true_theta))

    # Save outputs and plot
    if plot:
        timestamp = time.strftime('%b-%d-%Y_%H%M%S', time.localtime())
        np.savetxt("results/est_{}_{}.txt".format(method, timestamp), standardized_theta_estimates)

        fig, axs = plt.subplots(2)
        axs[0].plot(estimation_errs, '-o')
        axs[0].set_ylabel("Estimation Error")
        axs[0].set_yscale('log')

        axs[1].plot(simulation_errs, '-x')
        axs[1].set_ylabel("Simulation Error")
        fig.savefig("results/errs_{}_{}.png".format(method, timestamp))
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(y_test.reshape(-1), label="Target", color="green")
        line, = ax.plot(simulated_samples_list[0].reshape(-1), label="Estimate".format(0), color="orange")

        def animate(i):
            line.set_ydata(simulated_samples_list[i].reshape(-1))
            plt.suptitle('Iteration :{}'.format(i))

        ani = animation.FuncAnimation(fig, animate,
                                      np.arange(1, len(simulated_samples_list)),
                                      interval=500, blit=False)

        writergif = animation.PillowWriter(fps=5)
        ani.save("results/trajectories_{}_{}.gif".format(method, timestamp), writer=writergif)
        plt.close()

    return simulation_errs, estimation_errs

def inertia_estimation_matrix(y_train, y_test, true_theta, actions, params, method, plot=True):
    '''
    Learn the inertia matrix as a single matrix
    If method == "SPD", learn on SPD manifold
    If method == "SPD_euclid", learn on Euclidean manifold, then project
    If method == "SPD_chol", learn with Cholesky parameterization (on Euclidean manifold)
    '''
    valid_methods = ["SPD", "SPD_euclid", "SPD_chol"]
    assert method in valid_methods

    # Unpack variable parameters
    num_herd = params['num_herd']
    theta_bandwidth = params['theta_bandwidth']
    theta_reg = params['theta_reg']
    lr = params['lr']
    num_epochs = params['num_epochs']
    num_iters = params['num_iters']
    num_processors = params['num_processors']

    # Set fixed parameters
    y_manifold_name = "Euclidean"
    dims = 3 # The system is actually 3 dimensional
    numeric_backend = "pytorch"
    y_bandwidth = None
    adapt_y_bandwidth = True
    adapt_theta_bandwidth = False

    # Get environment
    seed = 1234 # All simulations should start the same way
    env = ENV()
    env.seed(seed)
    env.reset()

    # Set up the manifolds
    if method == "SPD":
        theta_manifold_name = "SPD"
        theta_manifold = geoopt.SymmetricPositiveDefinite()
        riemannian_opt = True
    elif method == "SPD_euclid":
        theta_manifold_name = "Euclidean"
        theta_manifold = geoopt.Euclidean()
        riemannian_opt = False
    elif method == "SPD_chol":
        theta_manifold_name = "Euclidean"
        theta_manifold = geoopt.Euclidean()
        riemannian_opt = False
    else:
        raise(ValueError("Unknown Method. Pick from : {}".format(valid_methods)))

    # Set up the kernels
    kernel_theta = Laplacian(bandwidth=theta_bandwidth, manifold=theta_manifold_name)
    kernel_y = Laplacian(bandwidth=y_bandwidth, manifold=y_manifold_name)

    # Set up stadardization function
    def standardize_thetas(theta_list):
        '''
        Convert a list of thetas (SPD matrices) into the standard form (3 +ve eigenvalues, and 4 quaternions)
        '''

        # First convert to SPD matrices if needed
        if method == "SPD":
            theta_list_std = theta_list
        elif method == "SPD_euclid":
            # Project onto SPD
            theta_list_std = spd_project_func(theta_list)
        elif method == "SPD_chol":
            # Convert cholesky form into full SPD matrices
            theta_list_std = torch.zeros((theta_list.shape[0], dims, dims))
            for idx in np.arange(theta_list.shape[0]):
                theta_list_std[idx, :] = chol_to_spd(vec_to_chol(theta_list[idx, :], dims))
        else:
            raise(ValueError("Unknown method. Pick from {}".format(valid_methods)))

        # Eigen decomposition
        # TODO: D will always be sorted. So we can't ever match something with arbitrary order
        D, U = torch.symeig(theta_list_std, eigenvectors=True)
        if torch.any(D < 0):
            warnings.warn("Some eigenvalues of the SPD matrix are negative. Flipping them to +ve, but check for other errors.")
            D = torch.abs(D)
        #TODO: Identify approprirate clipping range
        D = torch.clip(D, 1e-10, 5e2)

        ## Convert U matrices into quaternions
        Q = rotmat_to_quats(U)
        return torch.hstack([D, Q])

    # Set up the simulator fuction
    def observation_simulator(theta_list):
        '''
        For a given list of thetas (inertia matrices), produce one sample of resulting trajectories each
        We use the same series of actions for all simulations
        '''

        # First standardize thetas i.e. convert to [diagonals, quaternions] form
        theta_list_std = standardize_thetas(theta_list)
        if num_processors > 1:
            multi_proc_args = [[copy.deepcopy(env), theta, actions] for theta in theta_list_std]
            with Pool(min(num_processors, cpu_count() - 1)) as pool:
                results = pool.starmap(run_hand_simulation, multi_proc_args)
        else:
            results = []
            for sim_id in np.arange(theta_list_std.shape[0]):
                results.append(run_hand_simulation(env, theta_list_std[sim_id, :], actions))
        results = torch.from_numpy(np.stack(results))
        return results

    # Set up the prior theta sampler
    def prior_theta_sampler(num_samples, scaler=SCALER):
        '''
        Randomly sample SPD matrices
        '''
        torch.manual_seed(1234) # Reset seed for consistency of initial samples
        ## Generate random spd matrices
        spds = scaler * torch.rand((num_samples, num_lower_triu_elems(dims)))
        spds = torch.stack([chol_to_spd(vec_to_chol(x, dims=dims)) for x in spds])

        if method in ["SPD", "SPD_euclid"]:
            return spds
        elif method == "SPD_chol":
            # Convert SPDs into cholesky decomposition first
            samples = torch.cholesky(spds)
            samples_out = torch.zeros((samples.shape[0], num_lower_triu_elems(dims)))
            for idx in np.arange(samples.shape[0]):
                # Only keep the non-zero elements of the tril matrix
                samples_out[idx, :] = samples[idx, :][np.tril_indices(dims)]
            return samples_out
        else:
            raise(ValueError("Unknown method. Choose from {}".format(valid_methods)))

    # Define the distance function used to compare theta with true theta
    def theta_dist_func(theta_est, theta_true):
        '''
        Get distance between theta_est (estimated) and theta_true (True)
        theta_true: 7D vector. 1st 3 entries are +ve eigenvalues, last 4 entries are quaternions
        '''
        # Convert theta_list to the standard form
        theta_est_std = standardize_thetas(theta_est)

        # Convert theta_est_std in standard form to full SPD matrices
        theta_est_D = theta_est_std[:, 0:3]
        theta_est_R = quats_to_rotmat(theta_est_std[:,3:])
        theta_est_SPD = np.zeros((theta_est_std.shape[0], dims, dims))
        for idx in np.arange(theta_est_std.shape[0]):
            theta_est_SPD[idx,:,:] = theta_est_R[idx,:,:] @ np.diag(theta_est_D[idx,:]) @ theta_est_R[idx,:,:].T

        # Convert thea_true in standard form to full SPD matrices
        theta_true_D = theta_true[:, 0:3]
        theta_true_R = quats_to_rotmat(theta_true[:,3:])
        theta_true_SPD = np.zeros((theta_true.shape[0], dims, dims))
        for idx in np.arange(theta_true.shape[0]):
            theta_true_SPD[idx,:,:] = theta_true_R[idx,:,:] @ np.diag(theta_true_D[idx,:]) @ theta_true_R[idx,:,:].T

        # Get distances between theta_est and theta_true in terms of SPD geodesic distance
        return spd_dist_func(theta_est_SPD, theta_true_SPD)

        # # Get distances based on diag + quaternion form
        # dist_euclid = euclid_dist_func(theta_est_std[:, 0:3], theta_true[:, 0:3])
        # dist_quat = sphere_dist_func(theta_est_std[:, 3:], theta_true[:, 3:])
        # return dist_euclid + dist_quat

    # Set up KRABC using y_train
    krabc = KernelRecursiveABC(y_train, num_herd, prior_theta_sampler, observation_simulator,
                               theta_manifold, kernel_y, kernel_theta,
                               reg=theta_reg,
                               numeric_backend=numeric_backend,
                               adapt_y_bandwidth=adapt_y_bandwidth,
                               adapt_theta_bandwidth=adapt_theta_bandwidth,
                               riemannian_opt=riemannian_opt,
                               y_dist_func=euclid_dist_func,
                               true_theta=None,
                               theta_dist_func=theta_dist_func,
                               theta_project_func=None,
                               lr=lr,
                               num_epochs=num_epochs)
    theta_estimates, _ , _ = krabc.run_estimator(num_iters=num_iters)

    # Get estimation and simulation errors
    simulation_errs = []
    simulated_samples_list = []
    estimation_errs = []

    for idx in np.arange(theta_estimates.shape[0]):
        # estimation errors
        if method == "SPD_chol":
            estimation_err = theta_dist_func(theta_estimates[idx,:].reshape(1,-1), true_theta.reshape(1,-1))
        else:
            estimation_err = theta_dist_func(theta_estimates[idx, :].reshape(1, dims, dims), true_theta.reshape(1, -1))
        estimation_errs.append(estimation_err.item())

        # simulation errors
        simulated_samples = observation_simulator(torch.stack([theta_estimates[idx, :]]))
        simulation_err = torch.norm((simulated_samples - y_test).mean(dim=0))
        simulation_errs.append(simulation_err)
        simulated_samples_list.append(simulated_samples)
    if method == "SPD_chol":
        standardized_theta_estimates = standardize_thetas(theta_estimates[-1, :].reshape(1, -1))
    else:
        standardized_theta_estimates = standardize_thetas(theta_estimates[-1, :].reshape(1, dims, dims))

    print('Final Theta Estimate: {}'.format(standardized_theta_estimates))
    print('True Theta: {}'.format(true_theta))

    # Save outputs and plot
    if plot:
        timestamp = time.strftime('%b-%d-%Y_%H%M%S', time.localtime())
        np.savetxt("results/est_{}_{}.txt".format(method, timestamp), standardized_theta_estimates)

        fig, axs = plt.subplots(2)
        axs[0].plot(estimation_errs, '-o')
        axs[0].set_ylabel("Estimation Error")
        axs[0].set_yscale('log')

        axs[1].plot(simulation_errs, '-x')
        axs[1].set_ylabel("Simulation Error")
        fig.savefig("results/errs_{}_{}.png".format(method, timestamp))
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(y_test.reshape(-1), label="Target", color="green")
        line, = ax.plot(simulated_samples_list[0].reshape(-1), label="Estimate".format(0), color="orange")

        def animate(i):
            line.set_ydata(simulated_samples_list[i].reshape(-1))
            plt.suptitle('Iteration :{}'.format(i))

        ani = animation.FuncAnimation(fig, animate,
                                      np.arange(1, len(simulated_samples_list)),
                                      interval=500, blit=False)

        writergif = animation.PillowWriter(fps=5)
        ani.save("results/trajectories_{}_{}.gif".format(method, timestamp), writer=writergif)
        plt.close()

    return simulation_errs, estimation_errs


def inertia_estimate_experiment(y_trains, y_tests, true_thetas, actions, params, method, plot=False,
                                full_output=False):

    assert len(y_trains) == len(y_tests) == len(actions)

    if method in ["quat", "quat_euclid"]:
        exp_func = inertia_estimation_product_manifold
    elif method in ["SPD", "SPD_euclid", "SPD_chol"]:
        exp_func = inertia_estimation_matrix
    else:
        raise(ValueError("Unknown method"))

    sim_errs_full = []
    est_errs_full = []
    sim_errs_end = []
    for idx in np.arange(len(y_trains)):
        print("Running exp round {} of {}".format(idx, len(y_trains)))
        sim_errs, est_errs = exp_func(y_trains[idx], y_tests[idx], true_thetas[idx], actions[idx],
                       params, method, plot)
        sim_errs, est_errs = sim_errs, est_errs
        sim_errs_end.append(np.mean(sim_errs[-3:])) # Take average of sampling errs in the last three iterations
        sim_errs_full.append(sim_errs)
        est_errs_full.append(est_errs)

    sim_errs_end = np.array(sim_errs_end)

    if full_output:
        # Return the full sequences of simulation and estimation errors
        return sim_errs_full, est_errs_full
    else:
        # Return the mean end simulation error over datasets [useful for tuning]
        mean_err = np.mean(sim_errs_end)
        return mean_err

if __name__ == "__main__":

    # Load data
    # seeds = [0, 1, 2, 3, 4]
    seeds = [0]
    y_trains, y_tests, true_thetas, actions = [], [], [], []
    for seed in seeds:
        data = np.load('data/tune_data/data_seed{}.npy'.format(seed), allow_pickle=True).item()
        true_thetas.append(torch.from_numpy(data['true_theta']))
        y_trains.append(torch.from_numpy(data['observations']).reshape(1, -1))
        y_tests.append(torch.from_numpy(data['observations']).reshape(1, -1))
        actions.append(data['actions'])

    # Product manifold
    # params = {
    #     'num_herd': 20,
    #
    #     'theta_reg': 0.001,
    #     'lr': 0.01,
    #     'num_epochs': 100,
    #     'num_iters': 10,
    #     'num_processors': NUM_PROCESSORS
    # }

    # SPD Matrices
    params = {
        'num_herd': 20,
        'theta_bandwidth_diags': 1.0, # for product manifolds
        'theta_bandwidth_quats': 1.0, # for product manifolds
        'theta_bandwidth': 0.01, # for matrix manifolds
        'theta_reg': 1e-6,
        'lr': 0.1,
        'num_epochs': 100,
        'num_iters': 10,
        'num_processors': NUM_PROCESSORS
    }

    method = "SPD"
    res = inertia_estimate_experiment(y_trains, y_tests,
                                      true_thetas, actions,
                                      params, method,
                                      plot=True)
    print(res)


