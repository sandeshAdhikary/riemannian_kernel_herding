from src.kernels.kernels import RBF
from src.samplers.kernel_herding_combined import KernelHerder
import torch
import numpy as np
from tqdm import trange
from src.divergences.wasserstein import wasserstein_over_samples

torch.set_default_dtype(torch.double)

MANIFOLD_BACKENDS = ['pymanopt', 'geoopt']
NUMERIC_BACKENDS = ['numpy', 'pytorch']

class KernelRecursiveABC():
    def __init__(self, y, num_herd, prior_sampler, simulator, theta_manifold,
                 kernel_y, kernel_theta, reg=1e-10, numeric_backend="pytorch",
                 adapt_y_bandwidth=False, adapt_theta_bandwidth=False, riemannian_opt=True,
                 y_dist_func=None, true_theta=None, theta_dist_func=None, theta_project_func=None,
                 verbose=True, lr=1e-3, num_epochs=25):
        '''
        y: observations of shape (n_obs,dims)
        n_herd: Number of points to herd (including from prior sampler)
        prior_sampler: Prior samping function. Takes n_samples as input; outputs state samples from prior distribution
        simulator: Simulator function. Takes a list of thetas, and outputs an observation for each
        theta_manifold: The manifold (pymanopt,geoopt object) of thetas
        kernel_y: Kernel for observations y.
        kernel_theta: Kernel for parameters theta
            Both kernels should have a .bandwidth attribute that can be changed
            They should have .kernel_eval(X,Y) function that computes Gram matrix
        reg: Regularization for the kernel ridge regression. Defaults to 1e-10
        numeric_backend: Either "numpy" or "pytorch". All input data should match this
        adapt_y_bandwidth: If True, use median trick to adapt y_bandwidth at each iteration
        adapt_theta_bandwidth: If True, use median trick to adapt theta_bandwidth at each iteration
        riemannian_opt: If True, use Riemannian optimization for herding. If false, use Euclidean opt.
        y_dist_func: Optional; distance function for observations. Defaults to Eucildean
        true_theta: Optional; true theta to be estimated. If provided, compute errors wrt it
        theta_dist_func: Optional; distance function for theta. Defaults to Euclidean
        theta_project_func: Optional; projection function to apply on theta
        '''
        self.n_obs = y.shape[0]
        self.y = y
        self.num_herd = num_herd
        self.prior_sampler = prior_sampler
        self.simulator = simulator
        self.theta_manifold = theta_manifold
        self.kernel_y = kernel_y
        self.kernel_theta = kernel_theta
        self.reg = reg
        self.numeric_backend = numeric_backend
        self.true_theta = true_theta
        self.theta_project_func = theta_project_func
        self.adapt_y_bandwidth = adapt_y_bandwidth
        self.adapt_theta_bandwidth = adapt_theta_bandwidth
        self.riemannian_opt = riemannian_opt
        self.verbose = verbose

        # Check if product manifold
        try:
            self.num_manifolds = theta_manifold.n_manifolds
        except:
            self.num_manifolds = 1

        ## Hyperparameters for Riemannian adam
        self.riemannian_adam_lr = lr
        self.riemannian_adam_epochs = num_epochs

        if self.numeric_backend == "pytorch":
            self.nlib = torch
        elif self.numeric_backend == "numpy":
            self.nlib = np
        else:
            raise (ValueError("Unknown numeric_backend {}. Pick from {}".format(numeric_backend,
                                                                               NUMERIC_BACKENDS)))

        self.prior_thetas = self.prior_sampler(num_herd)
        self.numdims = len(self.prior_thetas.shape) - 1
        # If self.num_dims > 1, samples are not vectors

        if self.true_theta is not None:
            self.theta_dist_func = theta_dist_func
            if self.theta_dist_func is None:
                self.theta_dist_func = self.euclidean_dist_func
                print("theta_dist_func not provided. Using Euclidean distance instead")

        self.y_dist_func = y_dist_func
        if self.y_dist_func is None:
            self.y_dist_func = self.euclidean_dist_func
            print("y_dist_func not provided. Using Euclidean distance instead")

    def lstsq(self, A, B):
        if self.numeric_backend == "pytorch":
            return torch.lstsq(B, A)[0]
        elif self.numeric_backend == "numpy":
            return np.linalg.lstsq(A,B)[0]

    def run_estimator(self, num_iters):
        '''
        num_iters: Number of iterations to run krabc estimator
        '''

        theta_estimates = self.nlib.zeros(num_iters+1, *self.prior_thetas.shape[1:])
        theta_herd = self.prior_thetas
        theta_estimate = self.get_theta_estimate(theta_herd)
        theta_estimates[0, :] = theta_estimate
        if self.true_theta is not None:
            if self.verbose:
                print("\nEstimation Error: {:.4f}".format(self.estimation_err(theta_estimate)))

            est_errs = self.nlib.zeros(num_iters+1)
            est_errs[0] = self.estimation_err(theta_estimate)
        else:
            est_errs = None

        for iter in trange(num_iters, desc="KR-ABC progress"):
            # Generate observations from the simulator conditioned on theta_herd
            y_sampled = self.simulator(theta_herd)
            # Update the kernel bandwidths with median trick if needed
            # TODO: Allow adapt_bandwidth for product kernels
            if self.adapt_y_bandwidth:
                self.kernel_y.bandwidth = self.median_trick(y_sampled, self.y_dist_func)
            if self.adapt_theta_bandwidth:
                self.kernel_theta.bandwidth = self.median_trick(theta_herd, self.theta_dist_func)

            # Get gram matrix/vector for observations
            G_y = self.kernel_y.kernel_eval(y_sampled, y_sampled)
            k_y = self.kernel_y.kernel_eval(y_sampled, self.y)

            # Get weights for the posterior
            w = self.lstsq(G_y + self.num_herd*self.reg*self.nlib.eye(G_y.shape[0]),
                           k_y)
            if self.numeric_backend == "numpy":
                w = np.mean(w, axis=1).reshape(-1)
            else:
                w = torch.mean(w, dim=1).reshape(-1)

            # Herding
            kernel_herder = KernelHerder(theta_herd, w, self.kernel_theta, self.theta_manifold,
                                         numeric_backend=self.numeric_backend)

            theta_herd = kernel_herder.run_herding(self.num_herd, self.riemannian_opt,
                                                   lr=self.riemannian_adam_lr, num_epochs=self.riemannian_adam_epochs)
            theta_estimate = self.get_theta_estimate(theta_herd)
            theta_estimates[iter+1, :] = theta_estimate
            if self.true_theta is not None:
                if self.verbose:
                    print("\nEstimation Error: {:.4f}".format(self.estimation_err(theta_estimate)))

                est_errs[iter+1] = self.estimation_err(theta_estimate) if self.true_theta is not None else None




        # Get sampling errors
        # sampling_errs = self.sampling_err(theta_estimates)
        sampling_errs = None

        return theta_estimates, est_errs, sampling_errs

    def sampling_err(self, theta_est):
        '''
        Generate samples using theta_est (an array of estimates),
        and check how closely they match the true data distribution
        '''
        y_sampled = self.simulator(theta_est)
        # return wasserstein
        return wasserstein_over_samples(y_sampled, self.y, self.y_dist_func, step=1)

    def estimation_err(self, theta_est):
        '''
        If true_theta is given, compute estimation error for an estimate theta_est
        '''
        if self.true_theta is None:
            return None

        if self.numeric_backend == "pytorch":
            return torch.norm(theta_est - self.true_theta)/sum([*self.true_theta.shape])
        else:
            return np.linalg.norm(theta_est - self.true_theta)/sum([*self.true_theta.shape])

    def get_theta_estimate(self, theta_herd):
        '''
        Given a set of herded thetas, pick an estimated theta
        Right now, just returns the first herded point (the MAP estimate)
        '''

        return theta_herd[0, :]

    def euclidean_dist_func(self, X, Y):
        '''
        Default distance function to use in the median trick
        '''
        return RBF().get_pairwise_distances(X, Y, manifold="Euclidean")


    def median_trick(self, X, dist_func=None):
        '''
        Compute the bandwidth for kernel based on the median trick
        Reference: Section 2.2 in https://arxiv.org/pdf/1707.07269.pdf
        :param X: The data of shape (num_data, dims) or (num_data, dim1, dim2)
        dist_func: The distance function for pairwise distances
        :return:
        '''

        dist_func = dist_func if dist_func is not None else self.euclidean_dist_func

        # Get pairwise distances
        dists = dist_func(X, X).reshape(-1)

        # Compute median of the dists
        h = self.nlib.median(dists)

        # Get bandwidth
        nu = self.nlib.sqrt(h / 2.0)

        return nu

