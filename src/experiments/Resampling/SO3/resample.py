import numpy as np
import torch
# from geoopt import Stiefel
from pymanopt.manifolds import SpecialOrthogonalGroup
from src.kernels.kernels import Laplacian, RotationKernel
from src.samplers.kernel_herding_combined import KernelHerder
from src.samplers.optimal_transport import OptimalTransporter
from src.manifold_utils.ortho_utils import centroid as CayleyCentroid
from src.manifold_utils.so3_utils import geodesic
from src.divergences.wasserstein import wasserstein_over_samples
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

DIMS = 3
MANIFOLD_NAME = "SO3"
TRUE_MANIFOLD = SpecialOrthogonalGroup(DIMS)
NUMERIC_BACKEND = "numpy"
DIST_FUNC = geodesic
NUM_PROCESSORS = 10

def herd_resample(data_filepath, params, plot_path=None):

    # Set up experiment parameters
    riemannian_opt = params['riemannian_opt']
    num_data = params['num_data']
    num_resample = params['num_resample']
    adam_lr = params['adam_lr']
    adam_epochs = params['adam_epochs']
    bandwidth = params['bandwidth']

    assert num_resample >= 5, "Resample at least 5 to get good wasserstein distance values"

    # Load data
    data = np.load(data_filepath, allow_pickle=True).item()
    X = data['samples'][0:num_data, :]
    w = data['weights'][0:num_data]
    w = w / w.sum()

    # Set up the manifold
    manifold_name = MANIFOLD_NAME
    true_manifold = TRUE_MANIFOLD
    numeric_backend = NUMERIC_BACKEND

    # Set up evaluation params
    dist_func = DIST_FUNC

    # Run kernel herding
    kernel_manifold = manifold_name if riemannian_opt else "Euclidean"
    kernel = Laplacian(bandwidth=bandwidth, manifold=kernel_manifold)
    kernel_herder = KernelHerder(X, w, kernel, true_manifold,
                                 numeric_backend=numeric_backend)
    X_herd = kernel_herder.run_herding(num_resample, riemannian_opt,
                                       lr=adam_lr, num_epochs=adam_epochs)
    if torch.is_tensor(X_herd):
       X_herd = X_herd.detach().numpy()

    # Evaluate herded samples
    wass = wasserstein_over_samples(X_herd, X, dist_func=dist_func,
                                    x_weights=None, y_weights=w,
                                    step=5, min_idx=5)

    if plot_path:
        try:
            fig = plt.figure()
            plt.plot(wass, '-o')
            plt.ylabel("Wasserstein distance from target empirical distribution")
            plt.xlabel('Number of Samples')
            plt.yscale('log')
            fig.savefig(plot_path)
            plt.close()
        except:
            print("Could not save plot, check plot_path")

    return wass

def herd_resample_char(data_filepath, params, type, plot_path=None):
    '''
    Resample using kernel herding using characteristic kerne
    type = [1,2] Picks from the two characteristic kernels for SO(3)
    '''

    # Set up experiment parameters
    riemannian_opt = params['riemannian_opt']
    num_data = params['num_data']
    num_resample = params['num_resample']
    adam_lr = params['adam_lr']
    adam_epochs = params['adam_epochs']
    bandwidth = params['bandwidth']

    assert num_resample >= 5, "Resample at least 5 to get good wasserstein distance values"

    # Load data
    data = np.load(data_filepath, allow_pickle=True).item()
    X = data['samples'][0:num_data, :]
    w = data['weights'][0:num_data]
    w = w / w.sum()

    # Set up the manifold
    true_manifold = TRUE_MANIFOLD
    numeric_backend = NUMERIC_BACKEND

    # Set up evaluation params
    dist_func = DIST_FUNC

    # Run kernel herding
    kernel = RotationKernel(type=type, bandwidth=bandwidth)
    kernel_herder = KernelHerder(X, w, kernel, true_manifold,
                                 numeric_backend=numeric_backend)
    X_herd = kernel_herder.run_herding(num_resample, riemannian_opt,
                                       lr=adam_lr, num_epochs=adam_epochs)
    if torch.is_tensor(X_herd):
       X_herd = X_herd.detach().numpy()

    # Evaluate herded samples
    wass = wasserstein_over_samples(X_herd, X, dist_func=dist_func,
                                    x_weights=None, y_weights=w,
                                    step=5, min_idx=5)

    if plot_path:
        try:
            fig = plt.figure()
            plt.plot(wass, '-o')
            plt.ylabel("Wasserstein distance from target empirical distribution")
            plt.xlabel('Number of Samples')
            plt.yscale('log')
            fig.savefig(plot_path)
            plt.close()
        except:
            print("Could not save plot, check plot_path")

    return wass

def herd_resample_euclid(data_filepath, params, plot_path=None):

    # Set up experiment parameters
    riemannian_opt = False
    num_data = params['num_data']
    num_resample = params['num_resample']
    adam_lr = params['adam_lr']
    adam_epochs = params['adam_epochs']
    bandwidth = params['bandwidth']

    assert num_resample >= 5, "Resample at least 5 to get good wasserstein distance values"

    # Load data
    data = np.load(data_filepath, allow_pickle=True).item()
    X = data['samples'][0:num_data, :]
    w = data['weights'][0:num_data]
    w = w / w.sum()

    # Set up the manifold
    true_manifold = TRUE_MANIFOLD
    numeric_backend = NUMERIC_BACKEND

    # Set up evaluation params
    dist_func = DIST_FUNC

    # Run kernel herding
    kernel_manifold = "Euclidean"
    kernel = Laplacian(bandwidth=bandwidth, manifold=kernel_manifold)
    kernel_herder = KernelHerder(X, w, kernel, true_manifold,
                                 numeric_backend=numeric_backend)
    X_herd = kernel_herder.run_herding(num_resample, riemannian_opt,
                                       lr=adam_lr, num_epochs=adam_epochs)
    if torch.is_tensor(X_herd):
       X_herd = X_herd.detach().numpy()

    # Evaluate herded samples
    wass = wasserstein_over_samples(X_herd, X, dist_func=dist_func,
                                    x_weights=None, y_weights=w,
                                    step=5, min_idx=5)

    if plot_path:
        try:
            fig = plt.figure()
            plt.plot(wass, '-o')
            plt.ylabel("Wasserstein distance from target empirical distribution")
            plt.xlabel('Number of Samples')
            plt.yscale('log')
            fig.savefig(plot_path)
            plt.close()
        except:
            print("Could not save plot, check plot_path")

    return wass

def ot_resample(data_filepath, params, plot_path=None):
    '''
    Run resampling with optimal transport.
    First, one round of sampling with SIR with given weights
    Then, these new samples get optimally transported
    '''

    # Set up experiment parameters
    num_resample = params['num_resample']
    ot_reg = params['ot_reg']
    adam_lr = params['adam_lr']
    num_data = params['num_data']
    adam_epochs = params['adam_epochs'] # Used if using default centroid_fn

    assert num_resample >= 5, "Resample at least 5 to get good wasserstein distance values"

    # Load data
    data = np.load(data_filepath, allow_pickle=True).item()
    Xt = data['samples'][0:num_data, :]
    wt = data['weights'][0:num_data]
    wt = wt/wt.sum()

    # Generate Xs by resampling from X based on its weights
    sir_idxes = np.random.choice(np.arange(Xt.shape[0]),
                                 size=num_resample, p=wt, replace=True)
    Xs = Xt[sir_idxes, :]

    # Set up manifold
    dist_func = DIST_FUNC
    manifold = TRUE_MANIFOLD

    optimal_transporter = OptimalTransporter(Xs, Xt, wt, dist_func, ot_reg,
                                             manifold=manifold,
                                             adam_lr=adam_lr,
                                             adam_epochs=adam_epochs,
                                             num_processes=1)

    Xr = optimal_transporter.resample()

    # Before evaluating shuffle Xr since we'll be computing distances over samples
    shuffled_idx = np.random.permutation(np.arange(Xr.shape[0]))
    Xr = Xr[shuffled_idx, :]

    # Evaluate herded samples
    wass = wasserstein_over_samples(Xr, Xt, dist_func=dist_func,
                                    x_weights=None, y_weights=wt,
                                    step=5, min_idx=5)

    if plot_path:
        try:
            fig = plt.figure()
            plt.plot(wass, '-o')
            plt.ylabel("Wasserstein distance from target empirical distribution")
            plt.xlabel('Number of Samples')
            plt.yscale('log')
            fig.savefig(plot_path)
            plt.close()
        except:
            print("Could not save plot, check plot_path")

    return wass

def ot_resample_pf(data_filepath, params, plot_path=None):
    '''
    Run resampling with optimal transport.
    First, one round of sampling with SIR with given weights
    Then, these new samples get optimally transported
    '''

    # Set up experiment parameters
    num_resample = params['num_resample']
    ot_reg = params['ot_reg']
    adam_lr = params['adam_lr']
    num_data = params['num_data']
    adam_epochs = params['adam_epochs'] # Used if using default centroid_fn

    assert num_resample >= 5, "Resample at least 5 to get good wasserstein distance values"

    # Load data
    data = np.load(data_filepath, allow_pickle=True).item()
    Xs = data['samples'][0:num_data, :]
    ws = data['weights'][0:num_data]
    ws = ws/ws.sum()

    # Generate Xs by resampling from X based on its weights
    sir_idxes = np.random.choice(np.arange(Xs.shape[0]),
                                 size=num_resample, p=ws, replace=True)
    Xt = Xs[sir_idxes, :]

    # Set up manifold
    dist_func = DIST_FUNC
    manifold = TRUE_MANIFOLD

    optimal_transporter = OptimalTransporter(Xs, Xt, dist_func, ot_reg,
                                             ws=ws,
                                             wt=None,
                                             manifold=manifold,
                                             adam_lr=adam_lr,
                                             adam_epochs=adam_epochs,
                                             num_processes=1)

    Xr = optimal_transporter.resample()

    # Before evaluating shuffle Xr since we'll be computing distances over samples
    shuffled_idx = np.random.permutation(np.arange(Xr.shape[0]))
    Xr = Xr[shuffled_idx, :]

    # Evaluate herded samples
    wass = wasserstein_over_samples(Xr, Xs, dist_func=dist_func,
                                    x_weights=None, y_weights=ws,
                                    step=5, min_idx=5)

    if plot_path:
        try:
            fig = plt.figure()
            plt.plot(wass, '-o')
            plt.ylabel("Wasserstein distance from target empirical distribution")
            plt.xlabel('Number of Samples')
            plt.yscale('log')
            fig.savefig(plot_path)
            plt.close()
        except:
            print("Could not save plot, check plot_path")

    return wass

def ot_resample_pf_cayley(data_filepath, params, plot_path=None):
    '''
    Run resampling with optimal transport.
    This variant uses the Cayley transform method to compute centroid
    First, one round of sampling with SIR with given weights
    Then, these new samples get optimally transported
    '''

    # Set up experiment parameters
    num_resample = params['num_resample']
    ot_reg = params['ot_reg']
    adam_lr = params['adam_lr']
    num_data = params['num_data']
    adam_epochs = params['adam_epochs'] # Used if using default centroid_fn

    assert num_resample >= 5, "Resample at least 5 to get good wasserstein distance values"

    # Load data
    data = np.load(data_filepath, allow_pickle=True).item()
    Xs = data['samples'][0:num_data, :]
    ws = data['weights'][0:num_data]
    ws = ws/ws.sum()

    # Generate Xs by resampling from X based on its weights
    sir_idxes = np.random.choice(np.arange(Xs.shape[0]),
                                 size=num_resample, p=ws, replace=True)
    Xt = Xs[sir_idxes, :]

    # Set up manifold
    dist_func = DIST_FUNC
    manifold = TRUE_MANIFOLD

    # Define the Cayley centroid function
    centroid_epochs = adam_epochs
    centroid_fn = lambda x, w: CayleyCentroid(x, w, num_iters=centroid_epochs)

    optimal_transporter = OptimalTransporter(Xs, Xt, dist_func, ot_reg,
                                             ws=ws,
                                             wt=None,
                                             manifold=manifold,
                                             adam_lr=adam_lr,
                                             adam_epochs=adam_epochs,
                                             num_processes=1,
                                             centroid_fn=centroid_fn)

    Xr = optimal_transporter.resample()

    # Before evaluating shuffle Xr since we'll be computing distances over samples
    shuffled_idx = np.random.permutation(np.arange(Xr.shape[0]))
    Xr = Xr[shuffled_idx, :]

    # Evaluate herded samples
    wass = wasserstein_over_samples(Xr, Xs, dist_func=dist_func,
                                    x_weights=None, y_weights=ws,
                                    step=5, min_idx=5)

    if plot_path:
        try:
            fig = plt.figure()
            plt.plot(wass, '-o')
            plt.ylabel("Wasserstein distance from target empirical distribution")
            plt.xlabel('Number of Samples')
            plt.yscale('log')
            fig.savefig(plot_path)
            plt.close()
        except:
            print("Could not save plot, check plot_path")

    return wass

def resampling_experiment(data_params):
    '''
    input: [data_filepath, params] array
    '''
    data_filepath, params = data_params

    resampling_technique = params['resampling_technique']

    if resampling_technique == 'kernel-herding':
        resample_func = herd_resample
    elif resampling_technique == 'kernel-herding-char1':
        resample_func = lambda data_file, params, plot_path: herd_resample_char(data_file, params, type=1, plot_path=plot_path)
    elif resampling_technique == 'kernel-herding-euclid':
        resample_func = herd_resample_euclid
    elif resampling_technique == 'optimal-transport-pf':
        resample_func = ot_resample_pf
    elif resampling_technique == 'optimal-transport-pf-cayley':
        resample_func = ot_resample_pf_cayley
    else:
        raise(ValueError("Unknown resampling technique"))

    wass = resample_func(data_filepath, params, plot_path=None)
    print(wass)
    return wass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Tune hyperparamters for kernel herding or optimal transport')
    parser.add_argument('resampling_technique')
    args = parser.parse_args()
    resampling_technique = args.resampling_technique

    # seeds = np.arange(1, 5) #Skip seed0 since that was used for tuning
    # seeds = [6]
    seeds = range(1, 5)
    # seeds = [0]

    if resampling_technique == 'kernel-herding':
        params = {
            "resampling_technique": resampling_technique,
            "num_data": 1500,
            "riemannian_opt": True,
            "num_resample": 1500,
            "adam_lr": 0.1,
            "adam_epochs": 1000,
            "bandwidth": 5.0
        }
    elif resampling_technique == 'kernel-herding-char1':
        params = {
            "resampling_technique": resampling_technique,
            "num_data": 1500,
            "riemannian_opt": True,
            "num_resample": 1500,
            "adam_lr": None,
            "adam_epochs": 1000,
            "bandwidth": None
        }
    elif resampling_technique == 'kernel-herding-euclid':
        params = {
            "resampling_technique": resampling_technique,
            "num_data": 1500,
            "riemannian_opt": False,
            "num_resample": 1500,
            "adam_lr": 0.001,
            "adam_epochs": 1000,
            "bandwidth": 1.0
        }
    elif resampling_technique == 'optimal-transport-pf':
        params = {
            "resampling_technique": resampling_technique,
            "num_data": 1500,
            "num_resample": 1500,
            "ot_reg": 0.1,
            "adam_lr": 0.1,
            "adam_epochs": 1000,
        }
    elif resampling_technique == 'optimal-transport-pf-cayley':
        params = {
            "resampling_technique": resampling_technique,
            "num_data": 1500,
            "num_resample": 1500,
            "ot_reg": 0.05,
            "adam_lr": 0.0001,
            "adam_epochs": 1000,
        }
    else:
        raise(ValueError("Unknown resampling technique"))

    data_filepaths = ['data/weighted_samples_dims3_dataseed{}.npy'.format(seed) for seed in seeds]

    params_list = [params] * len(seeds)

    with Pool(min(NUM_PROCESSORS, cpu_count() - 1)) as pool:  # Use one less than total cpus to prevent freeze
        results = pool.map(func=resampling_experiment, iterable=zip(data_filepaths, params_list))

    print(results)

    results_dict = {
        'errs': results,
        'seeds': seeds,
        'params' : params
    }
    np.save('results/resampling_exp_errs_{}.npy'.format(resampling_technique), results_dict)
