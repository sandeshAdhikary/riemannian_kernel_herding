# import torch
# from geomloss import SamplesLoss
# import numpy as np
# from manifold_utils.circle_utils import project as project_circle
# from manifold_utils.spd_utils import geodesic as spd_geodesic
# from manifold_utils.spd_utils import project as project_spd
# from manifold_utils.spd_utils import isPD
#
#
# def circle_dist(X, Y):
#     ## Assumed input shapes: [1, num_samples, num_dims]
#     ## Output shape: [1, num_samples_X, num_samples_Y]
#     # geomloss expectes multi-batch data, we're going to just use a single batch
#
#     X_ = X.reshape(X.shape[1], X.shape[2])
#     Y_ = Y.reshape(Y.shape[1], Y.shape[2])
#
#     # first project data onto circle
#     X_ = project_circle(X_)
#     Y_ = project_circle(Y_)
#
#     dist = X_.matmul(Y_.t())
#
#     # Clip distances to make sure arccos() works
#     # exact values -1 and 1 sometimes cause gradient issues
#     dist = torch.clip(dist, -0.999999, 0.9999)
#     dist = torch.arccos(dist)
#
#     return dist.unsqueeze(0)
#
#
# def spd_dist(X, Y):
#     """
#     :param X: shape (1, num_data_x, dim^2)
#     :param Y: shape (1, num_data_y, dim^2)
#     :return:
#     """
#     # Convert vectorized matrices back to matrices before calling SPD geodesic
#     assert np.sqrt(X.shape[2]).is_integer()
#     assert np.sqrt(Y.shape[2]).is_integer()
#     assert np.sqrt(X.shape[2]) == np.sqrt(Y.shape[2])
#
#     dim = np.int(np.sqrt(X.shape[2]))
#
#     X_ = X.reshape(X.shape[1], dim, dim)
#     Y_ = Y.reshape(Y.shape[1], dim, dim)
#
#     X_ = X_ if isPD(X_) else project_spd(X_)
#     Y_ = Y_ if isPD(Y_) else project_spd(Y_)
#
#     dist = spd_geodesic(X_, Y_)
#
#     # Output should be of shape (1, num_data_x, num_data_y)
#     return dist.unsqueeze(0)
#
#
# def wasserstein_dist(X, Y, dist_func=None):
#
#     ## The sinkhorn loss interpolates between Wasserstein -> MMD
#     ## In geomloss, setting blur ~ 0 = Wasserstein, blur ~ infty = MMD
#
#     if dist_func is None:
#         # Use cost/dist-func: (1/p)*(|x-y|^p) for p = 2
#         wassterstein_sinkhorn = SamplesLoss(loss="sinkhorn",
#                                             p=2,
#                                             blur=0.05)
#     else:
#         wassterstein_sinkhorn = SamplesLoss(loss="sinkhorn",
#                                             cost=dist_func,
#                                             blur=0.05)
#
#
#     assert type(X) == type(Y)
#     assert X.ndim == Y.ndim
#
#     if torch.is_tensor(X):
#         dist = wassterstein_sinkhorn(X, Y)
#     else:
#         dist = wassterstein_sinkhorn(torch.from_numpy(X), torch.from_numpy(Y))
#
#     return dist.cpu().numpy()
#
#
# def wasserstein_over_samples(X, Y, manifold_type=None):
#     '''
#     X: generated samples
#     Y: target samples
#     Get MMD for all incremental point sets in X
#     '''
#
#     assert X.shape[-1] == Y.shape[-1], "Target and reference distributions must have same dimension"
#
#     if manifold_type is not None:
#         if manifold_type == "circle":
#             dist_func = circle_dist
#             wass_dists = [wasserstein_dist(X[0:idx, :], Y, dist_func) for idx in np.arange(1, X.shape[0])]
#         elif manifold_type == "spd":
#             dist_func = spd_dist
#             wass_dists = [wasserstein_dist(X[:, 0:idx, :], Y, dist_func) for idx in np.arange(1, X.shape[1])]
#         else:
#             raise(NotImplementedError("Unknown Manifold name. Only support 'circle' right now"))
#     else:
#         raise(NotImplementedError)
#
#
#     return np.array(wass_dists).reshape(-1)
#
