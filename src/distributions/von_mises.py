import numpy as np
# import autograd.numpy as np
import torch
from hyperspherical_vae.distributions import VonMisesFisher

class VonMises3D():
    '''
    The vonMises-Fisher distribution: Gaussian wrapped around the unit sphere
    '''

    def __init__(self, loc, conc):
        '''
        :param loc: array of mean points. shape: (1, num_dims)
        :param conc: array of concentrations. shape: (1, 1)
        '''
        self.loc = loc if torch.is_tensor(loc) else torch.from_numpy(loc)
        self.conc = conc if torch.is_tensor(conc) else torch.from_numpy(conc)
        self.centroid = self.loc.reshape(1, -1)
        self.dist = VonMisesFisher(self.loc,
                                   self.conc
                                   )

        self.loc = self.loc.double()
        self.conc = self.loc.double()

    def log_prob(self, x):
        if torch.is_tensor(x):
            return self.dist.log_prob(x).double()
        else:
            return self.dist.log_prob(torch.from_numpy(x)).double()

    def grad_log_p(self, x):
        '''
        :param x: data point
        :return:  grad(log(p(x)) i.e. the score function at x
        '''
        x_ = torch.autograd.Variable(x, requires_grad=True).double()

        dlog_p = torch.autograd.grad(
            self.log_prob(x_).sum(),
            x_,
        )[0]
        return dlog_p.double()

    def sample(self, num_samples):
        return self.dist.sample(num_samples)


class VonMisesMixture3D():
    '''
    Mixture of VomMises3D distributions
    '''

    def __init__(self, locs, concs, weights):
        self.locs = locs if torch.is_tensor(locs) else torch.from_numpy(locs)
        self.concs = concs if torch.is_tensor(concs) else torch.from_numpy(concs)
        self.weights = weights if torch.is_tensor(weights) else torch.from_numpy(weights)

        self.centroid = torch.mean(self.locs, dim=0).double()

        assert abs(sum(self.weights) - 1.) <= 1e-7, "Weights must sum to 1"

        self.num_components = len(self.weights)
        assert self.num_components == self.locs.shape[0]
        assert self.num_components == self.concs.shape[0]

        self.dims = self.locs.shape[-1]

        self.dists = []
        for idx in np.arange(self.num_components):
            loc = self.locs[idx, :, :].reshape(1, -1).double()
            conc = self.concs[idx].reshape(1, 1).double()
            self.dists.append(VonMisesFisher(loc, conc))

    def log_prob(self, x):

        log_prob = 0.
        for idx, dist in enumerate(self.dists):
            log_prob += self.weights[idx] * torch.exp(dist.log_prob(x))

        return torch.log(log_prob).double()

    def grad_log_p(self, x):
        x_ = torch.autograd.Variable(x, requires_grad=True)

        dlog_p = torch.autograd.grad(
            self.log_prob(x_).sum(),
            x_,
        )[0]
        return dlog_p.double()

    def sample(self, num_samples):

        samples = torch.zeros((num_samples, 1, self.dims)).double()
        for idx in np.arange(num_samples):
            # Pick a component with probs from weights
            component = np.random.choice(np.arange(self.num_components),
                                         p=self.weights)
            # Generate a sample from that component
            samples[idx, :, :] = self.dists[component].sample(1).double()

        return samples.double()