import numpy as np
from src.samplers.kernel_herding import run_herding as pymanopt_herder
from src.samplers.kernel_herding_geoopt import run_herding as geoopt_herder
from src.kernels.kernels import RBF, Laplacian
import pymanopt
from pymanopt.solvers.steepest_descent import SteepestDescent
from src.manifold_utils.sphere_utils import isSphere
from src.manifold_utils.sphere_utils import project as sphere_project
from pymanopt.manifolds import Sphere
import torch

def compute_centroid(manifold, points, weights=None):
    """
    Adapted from pymanopt.solvers.nelder_mead.compute_centroid; added weighting
    Compute the centroid of `points` on the `manifold` as Karcher mean.
    """
    num_points = len(points)

    weights = weights if weights is not None else np.ones(num_points)/num_points # equal weights if None

    if type(manifold) == Sphere:
        isValid = isSphere
        project = sphere_project
    else:
        raise(NotImplementedError("Only Sphere implemeted thus far"))

    points = points if isValid(points) else project(points)

    @pymanopt.function.Callable
    def objective(y):
        accumulator = 0
        for i in range(num_points):
            accumulator += (manifold.dist(y, points[i]) ** 2)*weights[i]
        return accumulator / 2

    @pymanopt.function.Callable
    def gradient(y):
        g = manifold.zerovec(y)
        for i in range(num_points):
            g -= manifold.log(y, points[i])
        return g

    # XXX: Manopt runs a few TR iterations here. For us to do this, we either
    #      need to work out the Hessian of the Karcher mean by hand or
    #      implement approximations for the Hessian to use in the TR solver as
    #      Manopt. This is because we cannot implement the Karcher mean with
    #      Theano, say, and compute the Hessian automatically due to dependence
    #      on the manifold-dependent distance function, which is written in
    #      numpy.
    solver = SteepestDescent(maxiter=15)
    problem = pymanopt.Problem(manifold, objective, grad=gradient, verbosity=0)
    return solver.solve(problem)


def euclidean_dist_func(X, Y):
    return RBF().get_pairwise_distances(X, Y, manifold="Euclidean")

def median_trick(X, dist_func=None):
    '''
    Compute the bandwidth for kernel based on the median trick
    Reference: Section 2.2 in https://arxiv.org/pdf/1707.07269.pdf
    :param X:
    :return:
    '''

    dist_func = dist_func if dist_func is not None else euclidean_dist_func

    # Get pairwise distances
    dists = dist_func(X, X).reshape(-1)

    # Compute median of the dists
    h = np.median(dists)

    # Get nu (Usually how bandwidth is defined)
    nu = np.sqrt(h/2.0)
    # nu = np.sqrt(h)

    # Get b (our RBF kernels are defined wrt this bandwidth)
    # b = np.sqrt(nu/2.0)

    # For scipy's RBF
    b = nu

    return b



def KernelRecursiveABC(y, n_iters, prior_sampler, simulator, kernel_y,
                       kernel_y_type, kernel_theta_type,optimization_type,bandwidth_scale,
                       n_samples, reg, true_manifold, true_theta=None,  y_dist_func=None,
                       theta_dist_func=None, project_func=None, herding_algo="ConjugateGradient",
                       backend="pymanopt", kernel_theta=None):
    '''
    :param y: (y_dim X N_y) Array of N y_dim-dimensional obersations
    :param prior_sampler: Function that samples from the prior distribution
    :param simulator: Function that samples from P(y|theta) given a theta
    :param kernel_theta_type: Type of kernel to use for thetas ["Euclidean","Sphere","SPD"]
    :param optimization_type: Type of optimization to use in herding ["Euclidean","Riemannian"]
    :param bandwidth: Bandwidth for kernel_theta used in herding
    :param n_iters: Number of iterations of kernel recursive ABC
    :param n_samples: Number of samples to draw from the prior, and from herding
    :param kernel_y: Kernel function on the observations
    :param reg: Regularization constant for KBR
    :param true_manifold: The true manifold over which \theta is defined (pymanopt manifold object)
    :param manifold: The manifold over which the kernel is defined in kernel herding
    :param kernel_type: Will define the distance function to use ["Euclidean","Sphere","SPD"]
    :param true_theta: optional. If provided, compute estimation errors wrt this true_theta
    :return:
    '''

    n_obs, dims = y.shape # Get the number of true observations

    # Get initial samples for theta from the prior
    theta_herd = prior_sampler(n_samples)

    # theta_estimates = np.zeros((n_iters, dims))
    theta_estimates = np.expand_dims(np.zeros(theta_herd.shape[1:]),0).repeat(n_iters,0)
    theta_herds = np.zeros(theta_herd.shape)
    theta_herds = theta_herds[np.newaxis, :].repeat(n_iters, axis=0)

    if true_theta is not None:
        if torch.is_tensor(theta_herd):
            # Use first theta as current estimate
            theta_est = theta_herd[0, :].unsqueeze(0)
            err = theta_dist_func(theta_est, true_theta)
            print("\t[Iteration 1 of {}]: Error = {}".format(n_iters, err))
        else:
            # Use first theta as current estimate
            theta_est = np.expand_dims(theta_herd[0, :],0)
        theta_estimates[0, :] = theta_est.squeeze(0)
        theta_herds[0, :] = theta_herd

    for iter in np.arange(n_iters-1):

        print("[KR ABC]: Iteration {} of {}".format(iter+1, n_iters))
        if torch.is_tensor(y):
            y_star = y.detach().numpy()
        else:
            y_star = y

        # Generate observations from the simulator conditioned on theta_herd
        theta_herd = theta_herd if project_func is None else project_func(theta_herd)
        y_sampled = simulator(theta_herd) # One sample for each theta

        # Get new bandwidth for kernels using median trick
        bandwidth_y = median_trick(y_sampled, dist_func=y_dist_func)
        # bandwidth_x = median_trick(theta_herd, dist_func=theta_dist_func) * bandwidth_scale
        bandwidth_x = bandwidth_scale

        # Set up the y kernel with new bandwidths
        kernel_y = Laplacian(bandwidth=bandwidth_y, manifold=kernel_y_type).kernel_eval

        # Get the gram matrix, and gram vector wrt the true y_star
        G_y = kernel_y(y_sampled, y_sampled)
        # k_y = kernel_y(y_sampled, y_star).mean(axis=1)
        k_y = kernel_y(y_sampled, y_star)

        print('Bandwidths: x = {:.3f}, y = {:.3f}'.format(bandwidth_x, bandwidth_y))
        print('|| G_y - I ||: {}'.format(np.linalg.norm(G_y - np.eye(G_y.shape[0]))))
        print('|| G_y - 1 ||: {}'.format(np.linalg.norm(G_y - np.ones(G_y.shape))))

        # Get weights for the posterior
        w = np.linalg.lstsq(G_y + n_samples*reg*np.eye(G_y.shape[0]),
                            k_y,
                            rcond=None)[0]

        # TODO: Should we average weights when we have multiple y*
        w = np.mean(w, axis=1).reshape(-1)
        if backend == "pymanopt":
            # Herd new points from the kernel mean estimate of the powered posterior
            theta_herd = pymanopt_herder(data=theta_herd,
                                     true_manifold=true_manifold,
                                     num_herd=n_samples,
                                     kernel_type=kernel_theta_type,
                                     optimization_type=optimization_type,
                                     bandwidth=bandwidth_x,
                                     data_weights=w,
                                     verbose=False,
                                     maxiter=100,
                                     opt_algo=herding_algo
                                     )
        elif backend == "geoopt":
            theta_herd = geoopt_herder(data=theta_herd,
                                       true_manifold=true_manifold,
                                       num_herd=n_samples,
                                       kernel_fn=kernel_theta.kernel_eval,
                                       optimization_type=optimization_type,
                                       data_weights=None,
                                       num_epochs=10, lr=1e-2)
        theta_herds[iter+1, :] = theta_herd
        
        # Use first hereded point (MAP estimate of mean map) as our estimate for theta
        theta_est = theta_herd[0, :]
        theta_estimates[iter+1, :] = theta_est
        
        if true_theta is not None:
            # err = y_dist_func(theta_est.reshape(1,-1), true_theta)
            norm = np.linalg.norm(true_theta)
            normalizer = norm if norm > 0 else 1
            if torch.is_tensor(true_theta):
                err = torch.norm(theta_est - true_theta)
            else:
                err = np.linalg.norm(theta_est - true_theta)
            # err = abs(theta_est - true_theta).mean()/normalizer
            print("\t[Iteration {} of {}]: Error = {}".format(iter, n_iters, err))

        
    return theta_estimates, theta_herds


