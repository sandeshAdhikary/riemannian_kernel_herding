import numpy as np
# import autograd.numpy as np
import torch
import pymanopt.manifolds as pymanifolds
import geoopt.manifolds as geomanifolds
from src.viz.circle import plot_probs as plot_probs_circle
from src.viz.sphere import plot_probs as plot_probs_sphere
from src.samplers.mcmc import emcee_sampler

class RiemannianGaussian():
    '''
    A Riemannian gaussian distribution
    '''

    def __init__(self, manifold, dims, mu, Sigma, backend='pymanopt'):
        '''
        manifold: a manifold object
        mu: mean
        Sigma: covariance
        '''
        self.backend = backend
        self.manifold = manifold
        self.mu = mu
        self.dims = dims
        self.Sigma = Sigma
        self.metric = torch.pinverse(self.Sigma) # Inverse of covariance matrix
        if backend == 'pymanopt':
            self.cov_logdet = np.log(np.linalg.det(self.Sigma))  # Log Determinant of the covairance matrix
        elif backend == 'geoopt':
            self.cov_logdet = torch.logdet(self.Sigma)# Log Determinant of the covairance matrix


    def log_prob(self, x):
        '''
        Get log probability
        '''

        if self.backend == 'geoopt':
            assert torch.is_tensor(x)

        log_prob = self.dims*np.log(2*np.pi) # TODO: What is dims for matrix manifolds?
        log_prob -= self.cov_logdet

        deltas = self.delta(self.mu, x)
        log_prob = log_prob.repeat(x.shape[0]) - self.mahalanobis_dist(deltas, self.metric)
        log_prob = 0.5 * log_prob.reshape(-1, 1)

        return log_prob

    def grad_log_p(self, x):
        if backend == 'pymanopt':
            raise(NotImplementedError("grad_log_p not implemented for backend=pymanopt"))
        elif backend == 'geoopt':
            x_ = torch.autograd.Variable(x, requires_grad=True)
            dlog_p = torch.autograd.grad(self.log_prob(x_).sum(),
                                         x_,
                                         )[0]
        else:
            raise(Exception("Unknown backend"))
        return dlog_p

    def sample(self, num_samples):
        pass

    def mahalanobis_dist(self, deltas, metric):
        '''
        Compute the squared Mahalanobis distances
        '''
        if self.backend == 'pymanopt':
            return np.diag(deltas @ metric @ deltas.T)
        elif self.backend == 'geoopt':
            return torch.diag(deltas.matmul(torch.matmul(metric, deltas.T)))

    def delta(self,mu,x):
        '''
        Compute the delta function for the squared Mahalanobis distance
        where squared Mahalanobis dist = -0.5 (delta(x,y)^T Sigma^{-1} delta(x,y))
        '''
        # TODO: avoid for loop here
        x = x.reshape(-1,mu.shape[1])
        if self.backend == 'pymanopt':
            #TODO: Just passing tensors mu, x doesn't do the right thing. Need to avoid for loop here
            delta = np.stack([self.manifold.log(mu,x[idx,:].reshape(1,-1)) for idx in np.arange(x.shape[0])]).squeeze(1)
        elif self.backend == 'geoopt':
            delta = self.manifold.logmap(mu, x)
        else:
            raise(NotImplementedError("Unsupported backend. Pick form [pymanopt, geoopt]"))
        # delta = self.manifold.log(mu.repeat(x.shape[0], 1), x)
        return delta


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    dims = 3
    backend = 'geoopt'
    if backend == 'pymanopt':
        manifold = pymanifolds.Sphere(dims)
    elif backend == 'geoopt':
        manifold = geomanifolds.Sphere()

    if dims == 2:
        # Circle:
        mu = np.array([1.,1.])
        Sigma = np.diag(np.array([1.0,1.0]))*1.0
        plot_fun = plot_probs_circle
    elif dims == 3:
        # Sphere:
        mu = np.array([1.0,1.0,1.0])
        Sigma = np.diag(np.array([1.0,1.0,1.0]))*0.001
        plot_fun = plot_probs_sphere
    else:
        raise (NotImplementedError("Cant plot higher than 3D!"))

    mu = mu / np.linalg.norm(mu)
    mu = torch.from_numpy(mu).reshape(1, -1)
    Sigma = torch.from_numpy(Sigma)
    gaussian = RiemannianGaussian(manifold, dims, mu, Sigma, backend=backend)
    plot_fun(gaussian.log_prob, gaussian.grad_log_p)

    x = torch.randn((1, dims))
    x = x / torch.norm(x)
    gaussian.grad_log_p(x)

    #
    # # Generate samples and plot histogram
    # samples = emcee_sampler(gaussian.log_prob,
    #              num_samples=100,
    #              ndim=dims)
    #
    # print('Done')



