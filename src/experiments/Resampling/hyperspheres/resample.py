import numpy as np
from geoopt import Sphere
from src.kernels.kernels import Laplacian
from src.samplers.kernel_herding_combined import KernelHerder
from src.samplers.optimal_transport import OptimalTransporter
from src.manifold_utils.sphere_utils import geodesic as sphere_dist
from src.divergences.wasserstein import wasserstein_over_samples
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

NUM_PROCESSORS = 1

def herd_on_hyperspheres(data_filepath, params, plot_path=None):

    # Set up experiment parameters
    riemannian_opt = True
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
    manifold_name = "Sphere"
    true_manifold = Sphere()
    numeric_backend = "pytorch"

    # Set up evaluation params
    dist_func = sphere_dist

    # Run kernel herding
    kernel_manifold = manifold_name if riemannian_opt else "Euclidean"
    kernel = Laplacian(bandwidth=bandwidth, manifold=kernel_manifold)
    kernel_herder = KernelHerder(X, w, kernel, true_manifold,
                                 numeric_backend=numeric_backend)
    X_herd = kernel_herder.run_herding(num_resample, riemannian_opt,
                                       lr=adam_lr, num_epochs=adam_epochs)
    X_herd = X_herd.numpy()

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

def herd_on_hyperspheres_euclid(data_filepath, params, plot_path=None):

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
    manifold_name = "Sphere"
    true_manifold = Sphere()
    numeric_backend = "pytorch"

    # Set up evaluation params
    dist_func = sphere_dist

    # Run kernel herding
    kernel_manifold = manifold_name if riemannian_opt else "Euclidean"
    kernel = Laplacian(bandwidth=bandwidth, manifold=kernel_manifold)
    kernel_herder = KernelHerder(X, w, kernel, true_manifold,
                                 numeric_backend=numeric_backend)
    X_herd = kernel_herder.run_herding(num_resample, riemannian_opt,
                                       lr=adam_lr, num_epochs=adam_epochs)
    X_herd = X_herd.numpy()

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

def ot_on_hyperspheres(data_filepath, params, plot_path=None):
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
    dist_func = sphere_dist
    manifold = Sphere()

    optimal_transporter = OptimalTransporter(Xs, Xt, dist_func, ot_reg,
                                             wt=wt,
                                             manifold=manifold,
                                             adam_lr=adam_lr,
                                             adam_epochs=adam_epochs)

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

def ot_on_hyperspheres_pf(data_filepath, params, plot_path=None):
    '''
    This is the particle filter version of ot_on_hypersphere
    i.e. using the same protocol as
    "Particle  Filtering  on  the  Stiefel  Manifold  with  Optimal  Transport"

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


    # Load data
    data = np.load(data_filepath, allow_pickle=True).item()
    Xs = data['samples'][0:num_data, :]
    ws = data['weights'][0:num_data]
    ws = ws/ws.sum()

    # assert num_resample == Xs.shape[0], "Number of resamples must be the same as number of sample provided"

    # Generate Xs by resampling from X based on its weights
    sir_idxes = np.random.choice(np.arange(Xs.shape[0]),
                                 size=Xs.shape[0], p=ws, replace=True)
    Xt = Xs[sir_idxes, :].copy()

    # Set up manifold
    dist_func = sphere_dist
    manifold = Sphere()

    optimal_transporter = OptimalTransporter(Xs, Xt, dist_func, ot_reg,
                                             ws=ws,
                                             wt=None,
                                             manifold=manifold,
                                             adam_lr=adam_lr,
                                             adam_epochs=adam_epochs)

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

def resampling_experiment(data_filepath):

    # Kernel Herding
    # herd_tune_res = np.load('tuning/kernel_herding.npy', allow_pickle=True).item()
    # best_idx = np.argmin(herd_tune_res['val_loss'])
    # herd_tune_loss = herd_tune_res['val_loss'][best_idx]
    # herd_tune_params = herd_tune_res['hyperparameters'][best_idx]
    herding_params = {
        "num_data": 1000,
        "num_resample": 1000,
        "adam_lr": 0.1,
        "adam_epochs": 100,
        "bandwidth": 10.0
    }
    wass_herd = herd_on_hyperspheres(data_filepath, herding_params, plot_path=None)
    herd_wasses.append(wass_herd)

    # Kernel Herding (Euclidean)
    # herd_tune_res = np.load('tuning/kernel_herding_euclid.npy', allow_pickle=True).item()
    # best_idx = np.argmin(herd_tune_res['val_loss'])
    # herd_tune_loss = herd_tune_res['val_loss'][best_idx]
    # herd_tune_params = herd_tune_res['hyperparameters'][best_idx]
    herding_params = {
        "num_data": 1000,
        "num_resample": 1000,
        "adam_lr": 0.1,
        "adam_epochs": 100,
        "bandwidth": 0.0001
    }
    wass_herd_euclid = herd_on_hyperspheres_euclid(data_filepath, herding_params, plot_path=None)
    herd_euclid_wasses.append(wass_herd_euclid)

    # Optimal Transport PF
    # ot_tune_res = np.load('tuning/optimal_transport_pf.npy', allow_pickle=True).item()
    # best_idx = np.argmin(ot_tune_res['val_loss'])
    # ot_tune_loss = ot_tune_res['val_loss'][best_idx]
    # ot_tune_params = ot_tune_res['hyperparameters'][best_idx]
    ot_params = {
        "num_data": 1000,
        "num_resample": 1000,
        "ot_reg": 0.1,
        "adam_lr": 0.1,
        "adam_epochs": 100,
    }
    wass_pf_ot = ot_on_hyperspheres_pf(data_filepath, ot_params, plot_path=None)
    ot_pf_wasses.append(wass_pf_ot)

    return herd_wasses, herd_euclid_wasses, ot_pf_wasses

if __name__ == "__main__":
    herd_wasses = []
    herd_euclid_wasses = []
    ot_pf_wasses = []
    # seeds = np.arange(1, 5)
    seeds = [1]

    data_filepaths = ['data/weighted_samples_sphere_dist_dims4_seed123_dataseed{}.npy'.format(seed) for seed in seeds]

    num_processors = NUM_PROCESSORS
    with Pool(min(num_processors, cpu_count() - 1)) as pool:  # Use one less than total cpus to prevent freeze
        results = pool.map(func=resampling_experiment, iterable=data_filepaths)
    results_dict = {
        'herding_errs': [res[0][0] for res in results],
        'herding_euclid_errs': [res[1][0] for res in results],
        'ot_pf_errs': [res[2][0] for res in results]
    }
    np.save('results/resampling_exp_errs.npy', results_dict)