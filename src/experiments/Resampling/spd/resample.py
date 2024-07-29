import numpy as np
from geoopt import SymmetricPositiveDefinite
import geoopt
from src.kernels.kernels import Laplacian
from src.samplers.kernel_herding_combined import KernelHerder
from src.samplers.optimal_transport import OptimalTransporter
from src.manifold_utils.spd_utils import geodesic
from src.divergences.wasserstein import wasserstein_over_samples
import matplotlib.pyplot as plt
from src.manifold_utils.spd_utils import num_lower_triu_elems, chol_to_spd, vec_to_chol
from multiprocessing import Pool, cpu_count
import torch

MANIFOLD_NAME = "SPD"
TRUE_MANIFOLD = SymmetricPositiveDefinite()
NUMERIC_BACKEND = "pytorch"
DIST_FUNC = geodesic
NUM_PROCESSORS = 8


def get_resamples(data_filepath, params, plot_path=None):
    # Set up experiment parameters
    num_data = params['num_data']
    num_resample = params['num_resample']
    adam_lr = params['adam_lr']
    adam_epochs = params['adam_epochs']
    method = params["method"]
    dist_func = DIST_FUNC

    assert num_resample >= 5, "Resample at least 5 to get good wasserstein distance values"

    # Load data
    data = np.load(data_filepath, allow_pickle=True).item()
    X = data['samples'][0:num_data, :]
    w = data['weights'][0:num_data]
    w = w / w.sum()

    if method == "OptimalTransport":
        ot_reg = params['ot_reg']
        # Generate Xt by resampling from X based on its weights
        Xs, ws = X, w
        sir_idxes = np.random.choice(np.arange(Xs.shape[0]),
                                     size=Xs.shape[0], p=ws, replace=True)
        Xt = Xs[sir_idxes, :].copy()

        # Set up manifold
        dist_func = DIST_FUNC
        manifold = SymmetricPositiveDefinite()
        optimal_transporter = OptimalTransporter(Xs, Xt,
                                                 dist_func, ot_reg,
                                                 ws=ws,
                                                 wt=None,
                                                 manifold=manifold,
                                                 adam_lr=adam_lr,
                                                 adam_epochs=adam_epochs)
        Xr = optimal_transporter.resample()

        # Before evaluating shuffle Xr since we'll be computing distances over samples
        shuffled_idx = np.random.permutation(np.arange(Xr.shape[0]))
        Xr = Xr[shuffled_idx, :]

    else:
        bandwidth = params['bandwidth']
        # kernel herding
        if method == "KernelHerding":
            # Run kernel herding
            manifold_name = "SPD"
            riemannian_opt = True
            true_manifold = SymmetricPositiveDefinite()
            kernel = Laplacian(bandwidth=bandwidth, manifold=manifold_name)
            kernel_herder = KernelHerder(X, w, kernel, true_manifold,
                                         numeric_backend=NUMERIC_BACKEND)
            Xr = kernel_herder.run_herding(num_resample, riemannian_opt,
                                           lr=adam_lr, num_epochs=adam_epochs)
            Xr = Xr.detach().numpy()
        elif method == "KernelHerdingEuclid":
            # Run kernel herding (Euclidean)
            manifold_name = "Euclidean"
            riemannian_opt = False
            true_manifold = geoopt.Euclidean()
            kernel = Laplacian(bandwidth=bandwidth, manifold=manifold_name)
            kernel_herder = KernelHerder(X, w, kernel, true_manifold,
                                         numeric_backend=NUMERIC_BACKEND)
            Xr = kernel_herder.run_herding(num_resample, riemannian_opt,
                                           lr=adam_lr, num_epochs=adam_epochs)
            Xr = Xr.detach().numpy()
        elif method == "KernelHerdingChol":
            # Run kernel herding using Cholesky parameterization
            dims = X.shape[1]
            X_chol_form = np.linalg.cholesky(X)
            X_chol = np.zeros((X_chol_form.shape[0], num_lower_triu_elems(dims)))
            for idx in np.arange(X_chol_form.shape[0]):
                # Only keep the flattened lower-triangular elements
                X_chol[idx, :] = X_chol_form[idx, :][np.tril_indices(3)]

            manifold_name = "Euclidean"
            true_manifold = geoopt.Euclidean()
            riemannian_opt = False

            kernel = Laplacian(bandwidth=bandwidth, manifold=manifold_name)
            # We herd around X_chol instead of X
            kernel_herder = KernelHerder(X_chol, w, kernel, true_manifold,
                                         numeric_backend=NUMERIC_BACKEND)
            Xr = kernel_herder.run_herding(num_resample, riemannian_opt,
                                           lr=adam_lr, num_epochs=adam_epochs)
            # Convert Xr from chol_vecs to cholesky matrices and then to full SPD
            Xr = chol_vecs_to_sdps(Xr, dims)

            if torch.is_tensor(Xr):
                Xr = Xr.detach().numpy()
        else:
            raise(ValueError("Unknown method"))

    # Evaluate the re-samples
    wass = wasserstein_over_samples(Xr, X, dist_func=dist_func,
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


def chol_vecs_to_sdps(chol_vecs, dims):
    '''
    Convert a cholesky vector back into a cholesky matrix
    '''
    tril_idxes = [[x, y] for (x, y) in zip(np.tril_indices(dims)[0], np.tril_indices(dims)[1])]
    out = np.zeros((chol_vecs.shape[0], dims, dims))
    for idx in np.arange(chol_vecs.shape[0]):
        for chol_idx, tril_idx in enumerate(tril_idxes):
            # Get cholesky matrix
            out[idx, tril_idx[0], tril_idx[1]] = chol_vecs[idx, chol_idx]
        # Get SPD matrix
        out[idx, :] = out[idx, :]@ out[idx, :].transpose()
    return out

if __name__ == "__main__":

    # methods: [KernelHerding, KernelHerdingEuclid, KernelHerdingChol, OptimalTransport]

    params = {
        "num_data": int(150),
        "method": "KernelHerdingChol",
        "num_resample": int(150),
        "adam_lr": 0.1,
        "adam_epochs": 100,
        "bandwidth": 0.001,
        "ot_reg": 0.01
    }

    # seeds = [1, 2, 3, 4, 5]
    seeds = [1,2]
    data_filepaths = ['data/weighted_samples_spd_dist_dims3_seed123_dataseed{}.npy'.format(seed) for seed in seeds]
    exp_inputs = [[data_filepath, params] for data_filepath in data_filepaths]

    num_processors = NUM_PROCESSORS
    with Pool(min(num_processors, cpu_count() - 1)) as pool:  # Use one less than total cpus to prevent freeze
        results = pool.starmap(func=get_resamples, iterable=exp_inputs)
    # np.save('results/resampling_exp_errs_{}.npy'.format(params['method']), results)


    wass = get_resamples(data_filepaths[0], params, plot_path=None)
    plt.plot(wass)
    plt.show()
    plt.savefig('resamples_{}'.format(params['method']))
    plt.close()