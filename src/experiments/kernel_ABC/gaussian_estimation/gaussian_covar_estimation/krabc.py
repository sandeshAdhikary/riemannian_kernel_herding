import geoopt
import torch
import numpy as np
from src.kernels.kernels import Laplacian
from src.manifold_utils.spd_utils import geodesic as spd_dist_func
from src.manifold_utils.spd_utils import chol_to_spd,vec_to_chol, num_lower_triu_elems
from src.manifold_utils.euclidean_utils import geodesic as euclid_dist_func
from src.manifold_utils.spd_utils import project as spd_project_func
from src.manifold_utils.spd_utils import isPD, project
from src.kr_abc.kr_abc_combined import KernelRecursiveABC
from torch.distributions.multivariate_normal import MultivariateNormal
from src.divergences.wasserstein import wasserstein_dist
from multiprocessing import Pool, cpu_count
import argparse
torch.set_default_dtype(torch.double)

NUM_SEEDS = 5
NUM_PROCESSORS = 1
PRIOR_SCALER = 1000 # Scales the covariances in the prior distribution by this amount

def covar_estimation_herd(y_train, y_test, params):

    # Unpack parameters
    riemannian_opt = True
    dims = params['dims']
    num_iters = params['num_iters']
    theta_bandwidth = params['theta_bandwidth']
    theta_reg = params['theta_reg']
    y_bandwidth = params['y_bandwidth']
    num_herd = params['num_herd']
    lr = params['lr']
    true_theta = params['true_theta']
    true_loc = params['true_loc']

    adapt_theta_bandwidth = True if theta_bandwidth is None else False
    adapt_y_bandwidth = True if y_bandwidth is None else False

    theta_manifold_name = "SPD"
    theta_manifold = geoopt.SymmetricPositiveDefinite()
    y_manifold_name = "Euclidean"

    # Set up the kernels
    kernel_theta = Laplacian(bandwidth=theta_bandwidth, manifold=theta_manifold_name)
    kernel_y = Laplacian(bandwidth=y_bandwidth, manifold=y_manifold_name)

    # Set up krabc parameters
    theta_dist_func = spd_dist_func # Used to compute errors between true_theta and estimated theta
    y_dist_func = euclid_dist_func # Used to compute errors between generated samples
    numeric_backend = "pytorch"
    theta_project_func = spd_project_func

    # Set up the simulator function
    def observation_simulator(theta_list, num_samples=1):
        '''
        Simulate observations of a given list of thetas
        '''
        dims = theta_list.shape[1]
        y_sampled = torch.zeros((theta_list.shape[0], num_samples, dims))
        theta_list_proj = project(theta_list) # Projection should not do anything if already PD
        for idx in np.arange(theta_list.shape[0]):
            target_dist = MultivariateNormal(true_loc, theta_list_proj[idx, :])
            y_sampled[idx, :] = target_dist.sample((num_samples,)).detach().reshape(num_samples, dims)
        if num_samples == 1:
            y_sampled = y_sampled.reshape(-1, dims)
        return y_sampled

    # Set up the prior theta sampler
    def prior_theta_sampler(num_samples, scaler=PRIOR_SCALER):
        samples = scaler * torch.rand((num_samples, num_lower_triu_elems(dims)))
        samples = torch.stack([chol_to_spd(vec_to_chol(x, dims=dims)) for x in samples])
        return samples

    # Set up KRABC using y_train
    krabc = KernelRecursiveABC(y_train, num_herd, prior_theta_sampler, observation_simulator,
                               theta_manifold, kernel_y, kernel_theta,
                               reg=theta_reg,
                               numeric_backend=numeric_backend,
                               adapt_y_bandwidth=adapt_y_bandwidth,
                               adapt_theta_bandwidth=adapt_theta_bandwidth,
                               riemannian_opt=riemannian_opt,
                               y_dist_func=y_dist_func,
                               true_theta=true_theta,
                               theta_dist_func=theta_dist_func,
                               theta_project_func=theta_project_func,
                               lr=lr)

    # Run KRABC using y_train
    theta_estimates, est_errs, _ = krabc.run_estimator(num_iters=num_iters)

    # Generate samples using final theta estimate
    simulated_samples = observation_simulator(torch.stack([theta_estimates[-1, :]]), num_samples=y_test.shape[0]).reshape(-1, dims)
    sampling_err = wasserstein_dist(simulated_samples, y_test, dist_func=y_dist_func)

    return sampling_err

def covar_estimation_herd_euclid(y_train, y_test, params):
    # Unpack parameters
    riemannian_opt = False
    dims = params['dims']
    num_iters = params['num_iters']
    theta_bandwidth = params['theta_bandwidth']
    theta_reg = params['theta_reg']
    y_bandwidth = params['y_bandwidth']
    num_herd = params['num_herd']
    lr = params['lr']
    true_theta = params['true_theta']
    true_loc = params['true_loc']

    adapt_theta_bandwidth = True if theta_bandwidth is None else False
    adapt_y_bandwidth = True if y_bandwidth is None else False

    theta_manifold = geoopt.SymmetricPositiveDefinite()
    theta_manifold_name = "Euclidean"
    y_manifold_name = "Euclidean"

    # Set up the kernels
    kernel_theta = Laplacian(bandwidth=theta_bandwidth, manifold=theta_manifold_name)
    kernel_y = Laplacian(bandwidth=y_bandwidth, manifold=y_manifold_name)

    # Set up krabc parameters
    theta_dist_func = euclid_dist_func # Used to compute errors between true_theta and estimated theta
    y_dist_func = euclid_dist_func # Used to compute errors between generated samples
    numeric_backend = "pytorch"
    theta_project_func = spd_project_func

    # Set up the simulator function
    def observation_simulator(theta_list, num_samples=1):
        '''
        Simulate observations of a given list of thetas
        '''
        dims = theta_list.shape[1]
        y_sampled = torch.zeros((theta_list.shape[0], num_samples, dims))

        theta_list_proj = theta_project_func(theta_list) # Projection should return as is if already PD
        for idx in np.arange(theta_list.shape[0]):
            target_dist = MultivariateNormal(true_loc, theta_list_proj[idx, :])
            y_sampled[idx, :] = target_dist.sample((num_samples,)).detach().reshape(num_samples, dims)
        if num_samples == 1:
            y_sampled = y_sampled.reshape(-1, dims)
        return y_sampled

    # Set up the prior theta sampler
    def prior_theta_sampler(num_samples, scaler=PRIOR_SCALER):
        samples = scaler * torch.rand((num_samples, num_lower_triu_elems(dims)))
        samples = torch.stack([chol_to_spd(vec_to_chol(x, dims=dims)) for x in samples])
        return samples

    # Set up KRABC using y_train
    krabc = KernelRecursiveABC(y_train, num_herd, prior_theta_sampler, observation_simulator,
                               theta_manifold, kernel_y, kernel_theta,
                               reg=theta_reg,
                               numeric_backend=numeric_backend,
                               adapt_y_bandwidth=adapt_y_bandwidth,
                               adapt_theta_bandwidth=adapt_theta_bandwidth,
                               riemannian_opt=riemannian_opt,
                               y_dist_func=y_dist_func,
                               true_theta=true_theta,
                               theta_dist_func=theta_dist_func,
                               theta_project_func=theta_project_func,
                               lr=lr)

    # Run KRABC using y_train
    theta_estimates, est_errs, _ = krabc.run_estimator(num_iters=num_iters)

    # Generate samples using final theta estimate
    simulated_samples = observation_simulator(torch.stack([theta_estimates[-1, :]]), num_samples=y_test.shape[0]).reshape(-1, dims)
    sampling_err = wasserstein_dist(simulated_samples, y_test, dist_func=y_dist_func)

    return sampling_err

def covar_estimation_herd_cholesky(y_train, y_test, params):
    # Unpack parameters
    riemannian_opt = False
    dims = params['dims']
    num_iters = params['num_iters']
    theta_bandwidth = params['theta_bandwidth']
    theta_reg = params['theta_reg']
    y_bandwidth = params['y_bandwidth']
    num_herd = params['num_herd']
    lr = params['lr']
    true_loc = params['true_loc']
    # Get true_theta and convert to cholesky vector
    true_theta = torch.cholesky(params['true_theta'])
    true_theta = true_theta[np.tril_indices(true_theta.shape[1])]

    adapt_theta_bandwidth = True if theta_bandwidth is None else False
    adapt_y_bandwidth = True if y_bandwidth is None else False

    theta_manifold_name = "Euclidean"
    theta_manifold = geoopt.Euclidean()
    y_manifold_name = "Euclidean"

    # Set up the kernels
    kernel_theta = Laplacian(bandwidth=theta_bandwidth, manifold=theta_manifold_name)
    kernel_y = Laplacian(bandwidth=y_bandwidth, manifold=y_manifold_name)

    # Set up krabc parameters
    theta_dist_func = euclid_dist_func
    y_dist_func = euclid_dist_func
    numeric_backend = "pytorch"
    theta_project_func = None

    def observation_simulator(theta_list, num_samples=1):
        '''
        Simulate observations of a given list of thetas
        '''
        y_sampled = torch.zeros((theta_list.shape[0], num_samples, dims))
        for idx in np.arange(theta_list.shape[0]):
            covar = chol_to_spd(vec_to_chol(theta_list[idx, :], dims))
            target_dist = MultivariateNormal(true_loc, covar)
            y_sampled[idx, :] = target_dist.sample((num_samples,)).detach().reshape(num_samples, dims)
        if num_samples == 1:
            y_sampled = y_sampled.reshape(-1, dims)
        return y_sampled

    # Set up the prior theta sampler
    def prior_theta_sampler(num_samples, scaler=PRIOR_SCALER):
        samples = scaler*torch.rand((num_samples, num_lower_triu_elems(dims)))
        return samples

    # Set up KRABC using y_train
    krabc = KernelRecursiveABC(y_train, num_herd, prior_theta_sampler, observation_simulator,
                               theta_manifold, kernel_y, kernel_theta,
                               reg=theta_reg,
                               numeric_backend=numeric_backend,
                               adapt_y_bandwidth=adapt_y_bandwidth,
                               adapt_theta_bandwidth=adapt_theta_bandwidth,
                               riemannian_opt=riemannian_opt,
                               y_dist_func=y_dist_func,
                               true_theta=true_theta,
                               theta_dist_func=theta_dist_func,
                               theta_project_func=theta_project_func,
                               lr=lr)

    # Run KRABC using y_train
    theta_estimates, est_errs, _ = krabc.run_estimator(num_iters=num_iters)

    # Generate samples using final theta estimate
    simulated_samples = observation_simulator(torch.stack([theta_estimates[-1, :]]), num_samples=y_test.shape[0]).reshape(-1, dims)
    sampling_err = wasserstein_dist(simulated_samples, y_test, dist_func=y_dist_func)

    return sampling_err

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run krabc for gaussian covarianve estimation')
    parser.add_argument('exp_type')
    args = parser.parse_args()
    exp_type = args.exp_type

    # Load dataset
    data = np.load("data/covar_est_data_dims3.npy", allow_pickle=True).item()
    num_seeds = NUM_SEEDS
    num_processors = NUM_PROCESSORS

    if exp_type == 'kernel-herding':
        exp_func = covar_estimation_herd
        params = {
            'dims': data['true_theta'].shape[0],
            'num_iters': 25,
            'theta_bandwidth': 0.01,
            'theta_reg': 1e-09,
            'y_bandwidth': 0.1,
            'num_herd': 100,
            'lr': 0.1, # adam LR
            'true_theta': data['true_theta'],
            'true_loc': data['true_loc']
        }
    elif exp_type == 'kernel-herding-euclid':
        exp_func = covar_estimation_herd_euclid
        params = {
            'dims': data['true_theta'].shape[0],
            'num_iters': 25,
            'theta_bandwidth': None,
            'theta_reg': 1.0,
            'y_bandwidth': 0.1,
            'num_herd': 100,
            'lr': 0.001, # adam LR
            'true_theta': data['true_theta'],
            'true_loc': data['true_loc']
        }
    elif exp_type == 'kernel-herding-cholesky':
        exp_func = covar_estimation_herd_cholesky
        params = {
            'dims': data['true_theta'].shape[0],
            'num_iters': 25,
            'theta_bandwidth': None,
            'theta_reg': 1.0,
            'y_bandwidth': 0.1,
            'num_herd': 100,
            'lr': 0.001, # adam LR
            'true_theta': data['true_theta'],
            'true_loc': data['true_loc']
        }
    else:
        raise(ValueError("Unknown experiment type"))

    # Load training and test observations
    y_trains = data['y_train']
    y_tests = data['y_test']
    y_trains = [y_trains[idx, :] for idx in np.arange(min(num_seeds, y_trains.shape[0]))]
    y_tests = [y_tests[idx, :] for idx in np.arange(min(num_seeds, y_tests.shape[0]))]

    with Pool(min(num_processors, cpu_count() - 1)) as pool:  # Use one less than total cpus to prevent freeze
        results = pool.starmap(exp_func, zip(y_trains, y_tests, np.array([params]*len(y_trains))))

    np.save('results/krabc-results-{}.npy'.format(exp_type), results)