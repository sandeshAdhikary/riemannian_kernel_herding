import numpy as np
from emcee import EnsembleSampler
import matplotlib.pyplot as plt
import geoopt
import torch
from mici import systems, integrators, samplers
from src.manifold_utils.sphere_utils import isSphere
from src.manifold_utils.spd_utils import isPD

def mcmc_mici(log_prob_fn, manifold_type, sample_shape, seed=0,backend='pymanopt'):
    '''
    constraint_fn(x) should return True if x is on manifold
    Code following Torus example in mici docs
    manifold must be a geoopt manifold
    '''

    if manifold_type == 'Sphere':
        constraint_fn = isSphere
    elif manifold_type == 'SPD':
        constraint_fn = isPD
    else:
        raise(Exception("Unknown Manifold"))

    constraint_fn = lambda x: 1.0 if constraint_fn else 0.0
    neg_log_prob_fn = lambda x: -log_prob_fn(x)

    # Specify constrained Hamiltonian system with default identity metric
    system = systems.DenseConstrainedEuclideanMetricSystem(neg_log_prob_fn, constraint_fn)

    # System is constrained therefore use constrained leapfrog integrator
    integrator = integrators.ConstrainedLeapfrogIntegrator(system)
    rng = np.random.default_rng(seed=seed)
    sampler = samplers.DynamicMultinomialHMC(system, integrator, rng)
    # sampler = samplers.StaticMetropolisHMC(system, integrator, rng, n_step=100)
    # Get initial point
    if backend == 'pymanopt':
        q_init = [manifold.rand()]*4
    elif backend == 'geoopt':
        q_init = manifold.random_uniform((sample_shape)).detach().numpy()

    # Sample 4 chains in parallel with 500 adaptive warm up iterations in which the
    # integrator step size is tuned, followed by 2000 non-adaptive iterations
    final_states, _, _ = sampler.sample_chains_with_adaptive_warm_up(
        n_warm_up_iter=500, n_main_iter=2000, init_states=q_init, n_process=1)

    return final_states
# Set up distribution as a torch NNs for geoopt samplers
class distNN(torch.nn.Module):
    def __init__(self, log_prob_func, shape, manifold=None):
        '''
        log_prob_func: torch function that (differentiably) computes log_prob
        shape: shape of elements of manifold. e.g. sphere (1,2)
        manifold: a geoopt manifold object, or None
        '''
        super().__init__()
        self.log_prob_func = log_prob_func

        if manifold is None:
            self.x = torch.nn.Parameter(torch.randn(shape))
        else:
            self.x = geoopt.ManifoldParameter(
                manifold.random_uniform(shape),
                manifold=manifold
            )

    def forward(self):
        return self.log_prob_func(self.x).sum()

def rhmc_geoopt(log_prob_func, n_samples, manifold=None, shape=None,
         epsilon=1e-3, n_steps=10, n_burn=100):
    '''
    Wrapper for Riemannain Hamiltonian Monte Carlo in geoopt
    log_prob_func: torch function that (differentiably) computes log_prob
    shape: shape of elements of manifold. e.g. sphere (1,2)
    manifold: a geoopt manifold object, or None
    '''

    # Set up sampler
    nd = distNN(log_prob_func, manifold=manifold, shape=shape)
    sampler = geoopt.samplers.rhmc.RHMC(nd.parameters(),
                                        epsilon=epsilon,
                                        n_steps=n_steps)
    # burn, baby burn
    for _ in range(n_burn):
        sampler.step(nd)

    points = []
    sampler.burnin = False
    for _ in range(n_samples):
        sampler.step(nd)
        points.append(nd.x.detach().numpy().copy())
    points = np.asarray(points)
    return points

def emcee_sampler(log_prob, num_samples, ndim,
                 nwalkers=10,
                 nsteps=5000,
                 burn=100,
                 init_samples=None):
    '''
    '''

    sampler = EnsembleSampler(nwalkers, ndim, log_prob)

    if nsteps*nwalkers < num_samples:
        nsteps = np.int(num_samples/nwalkers + 10)

    if init_samples is None:
        init_samples = np.random.random((nwalkers, ndim))

    # burn-in first few steps
    state = sampler.run_mcmc(init_samples, burn)
    sampler.reset()

    sampler.run_mcmc(state, nsteps=nsteps)

    return sampler.get_chain(flat=True)[0:num_samples, :]

if __name__ == "__main__":
    ## Test emcee
    # np.random.seed(42)
    # ndim = 5
    # mu = np.random.rand(ndim)
    #
    # cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
    # cov = np.triu(cov)
    # cov += cov.T - np.diag(cov.diagonal())
    # cov = np.dot(cov, cov)
    #
    # def log_prob(x):
    #     diff = x - mu
    #     return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    #
    # num_samples = 5000
    # samples = emcee_sampler(log_prob, num_samples, ndim,
    #                         nwalkers=32,
    #                         nsteps=1000,
    #                         burn=100,
    #                         init_samples=None)
    #
    # plt.hist(samples[:, 0], 100, color="k", histtype="step")
    # plt.xlabel(r"$\theta_1$")
    # plt.ylabel(r"$p(\theta_1)$")
    # plt.gca().set_yticks([])
    # plt.show()

    ## Test RHMC
    from src.viz.circle import plot_probs as plot_probs_circle
    from src.viz.sphere import plot_probs as plot_probs_sphere
    import pymanopt.manifolds as pymanifolds
    import geoopt.manifolds as geomanifolds
    from src.distributions.riemannian_gaussian import RiemannianGaussian
    #
    dims = 3
    backend = 'pymanopt'
    if backend == 'pymanopt':
        manifold = pymanifolds.Sphere(dims)
    elif backend == 'geoopt':
        manifold = geomanifolds.Sphere()

    if dims == 2:
        # Circle:
        mu = np.array([1., 1.])
        Sigma = np.diag(np.array([1.0, 1.0]))*1.0
        plot_fun = plot_probs_circle
    elif dims == 3:
        # Sphere:
        mu = np.array([1.0, 1.0, 1.0])
        Sigma = np.diag(np.array([1.0, 1.0, 1.0]))*0.001
        plot_fun = plot_probs_sphere
    else:
        raise (NotImplementedError("Cant plot higher than 3D!"))

    mu = mu / np.linalg.norm(mu)
    mu = torch.from_numpy(mu).reshape(1, -1)
    Sigma = torch.from_numpy(Sigma)
    gaussian = RiemannianGaussian(manifold, dims, mu, Sigma, backend=backend)
    samples = mcmc_mici(gaussian.log_prob, 'Sphere', mu.shape, seed=0)

    # samples = rhmc_geoopt(gaussian.log_prob, n_samples=100,
    #      manifold=manifold, shape=mu.shape,
    #      epsilon=1e-3, n_steps=10, n_burn=100)

