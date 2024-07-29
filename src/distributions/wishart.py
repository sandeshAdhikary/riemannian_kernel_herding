import autograd.numpy as np
import torch
from gpytorch.priors.wishart_prior import InverseWishartPrior
from scipy.stats import invwishart
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class InverseWishart():
    """
    Wrapper around gpytorch's inverse wishart prior

    Wishart prior over n x n positive definite matrices
    pdf(Sigma) ~ |Sigma|^(nu - n - 1)/2 * exp(-0.5 * Trace(K^-1 Sigma))
    where nu > n - 1 are the degrees of freedom and K > 0 is the p x p scale matrix
    Reference: A. Shah, A. G. Wilson, and Z. Ghahramani. Student-t Processes as
        Alternatives to Gaussian Processes. ArXiv e-prints, Feb. 2014.
    """

    def __init__(self, nu, K):
        self.nu = nu
        self.K = K if torch.is_tensor(K) else torch.from_numpy(K)
        self.dim = K.shape[0]
        self.dist = InverseWishartPrior(nu, self.K)
        # TODO: self.centroid ?

    def log_prob(self, x):
        '''
        :param x: data point
        :return: log(p(x)) i.e. log of the PDF at x
        '''
        if torch.is_tensor(x):
            return self.dist.log_prob(x).double()
        else:
            return self.dist.log_prob(torch.from_numpy(x)).double()

    def grad_log_p(self, x):
        """
        :param x: data point
        :return:  grad(log(p(x)) i.e. the score function at x
        """

        x_ = x if torch.is_tensor(x) else torch.from_numpy(x)

        x_ = torch.autograd.Variable(x_, requires_grad=True)

        dlog_p = torch.autograd.grad(
            self.log_prob(x_).sum(),
            x_,
        )[0]
        return dlog_p.double()

    def sample(self, num_samples):
        """
        :param num_samples:
        :return: Generate random samples from the distribution
        :output: (num_samples, self.dim, self.dim)
        """
        ## GPyTorch's Wishart distribution does not seem to have a sampler
        ## For now, use scipy's Wishart
        out = invwishart.rvs(df=self.nu, size=num_samples, scale=self.K)
        out = torch.from_numpy(out)
        return out.double()

if __name__ == "__main__":
    nu = 3
    K = 1.0*torch.eye(2)
    dist = InverseWishart(nu, K)
    test = dist.sample(500)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    max_lim = 5
    ax.set_xlim(-1, max_lim)
    ax.set_ylim(-1, max_lim)
    ax.set_zlim(0, max_lim)
    angle = -60
    ax.view_init(0, angle)

    # Only keep points that fit in the axis
    test = test[np.linalg.norm(test, axis=(1, 2)) < max_lim, :, :]

    probs = np.exp(dist.log_prob(test).numpy())
    x = test[:, 0, 0].numpy()
    y = test[:, 0, 1].numpy()
    z = test[:, 1, 1].numpy()


    ax.scatter(x, y, z,
               c=probs,
               # cmap = cm.jet,
               cmap = cm.Blues_r,
               s=20, alpha=0.5)

    fig.tight_layout()

    plt.show()