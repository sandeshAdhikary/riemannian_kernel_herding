import autograd.numpy as np
# from manifold_utils.spd_utils import geodesic as spd_dist
from src.manifold_utils.sphere_utils import geodesic as sphere_dist
from src.manifold_utils.spd_utils import geodesic as spd_dist
from src.manifold_utils.euclidean_utils import geodesic as euclidean_dist
from src.manifold_utils.sym_utils import geodesic as sym_dist
from src.manifold_utils.rotation_utils import geodesic as rot_dist
from src.manifold_utils.rotation_utils import geodesic_det as rot_det_dist
from src.manifold_utils.oblique_utils import geodesic as oblique_dist
from src.manifold_utils.ortho_utils import geodesic as ortho_dist

from ot import emd2
from tqdm import tqdm
import matplotlib.pyplot as plt

def wasserstein_dist(X, Y, M = None, x_weights=None, y_weights=None,dist_func=None):
    """
    :param X:
    :param Y:
    :return:
    """

    if M is None:
        # Need to compute the distance metric matrix
        assert dist_func is not None
        M = np.ascontiguousarray(dist_func(X, Y))

    # If weights not provided, define uniform weights for all data points
    x_weights = 1.0*(np.ones((X.shape[0],)) / X.shape[0]) if x_weights is None else x_weights
    y_weights = 1.0*(np.ones((Y.shape[0],)) / Y.shape[0]) if y_weights is None else y_weights


    return emd2(x_weights, y_weights, M, numItermax=100000000)

# def wasserstein_over_samples(X,Y,manifold_type=None, step=50, min_idx=1):
#
#     if manifold_type == "Sphere":
#         dist_func = sphere_dist
#     elif manifold_type == "SPD":
#         dist_func = spd_dist
#     elif manifold_type == "Euclidean":
#         dist_func = euclidean_dist
#     elif manifold_type == "Symmetric":
#         dist_func = sym_dist
#     elif manifold_type == "Rotation":
#         dist_func = rot_dist
#     elif manifold_type == "RotationDet":
#         dist_func = rot_det_dist
#     elif manifold_type == "Oblique":
#         dist_func = oblique_dist
#     elif manifold_type == "Orthogonal":
#         dist_func = ortho_dist
#     else:
#         raise(NotImplementedError("Unknown manifold: only Sphere and SPD implemented"))
#
#     # Compute the distance metric matrix
#     M = np.ascontiguousarray(dist_func(X, Y))
#
#     # wasserstein_dist(X[0:10, :], Y, M[0:10, :])
#     # Loop over the samples in X and compute a list of wasserstein distances
#     wass_dists = [wasserstein_dist(X[0:idx, :], Y, M[0:idx,:]) for idx in tqdm(np.arange(min_idx, X.shape[0], step))]
#
#     return wass_dists

def wasserstein_over_samples(X,Y,dist_func=None, x_weights=None, y_weights=None, step=50, min_idx=1):

    # Compute the distance metric matrix
    M = np.ascontiguousarray(dist_func(X, Y))

    # Loop over the samples in X and compute a list of wasserstein distances
    wass_dists = [wasserstein_dist(X[0:idx, :], Y, M[0:idx, :], x_weights, y_weights) for idx in tqdm(np.arange(min_idx, X.shape[0], step))]

    return wass_dists

if __name__ == "__main__":
    np.random.seed(0)

    # Circle test
    # from distributions.von_mises import VonMises
    # print("Running test for circles...")
    # mu = np.pi * (1 / 2)  # Location (mean)
    # conc = 1.0  # Concentration
    # num_data = 1000
    # num_samples = 500
    # dist = VonMises(loc=mu, conc=conc)
    # data = dist.sample(num_data)
    #
    # X = dist.sample(num_samples)
    # dists = wasserstein_over_samples(X, data, manifold_type="circle")
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(dists)
    # ax.set_xlabel("Num Samples")
    # ax.set_ylabel("Wasserstein Distance (Circle)")
    # ax.set_yscale("log")
    # plt.show()
    # plt.close()



    # ## SPD test
    from src.distributions.wishart import Wishart
    print("Running test for SPD...")
    dims = 2
    nu = 4
    num_data = 1000
    num_samples = 500
    K = np.eye(dims) * (1 / dims)
    dist = Wishart(nu, K)
    data = dist.sample(num_data).numpy()

    X = dist.sample(num_samples).numpy()
    dists = wasserstein_over_samples(X, data, manifold_type="spd")
    fig, ax = plt.subplots(1, 1)
    ax.plot(dists)
    ax.set_xlabel("Num Samples")
    ax.set_ylabel("Wasserstein Distance (SPD)")
    ax.set_yscale("log")
    plt.show()
    plt.close()

