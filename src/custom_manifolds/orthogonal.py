import autograd.numpy as np
from pymanopt.manifolds import Stiefel
from src.manifold_utils.ortho_utils import geodesic as ortho_dist

class Orthogonal(Stiefel):
    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        # Initialize Stiefel manifold for square matrices (i.e. orthogonal matrices)
        super().__init__(n, n, k=1)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def dist(self, x, y):
        return ortho_dist(x.reshape(1,self._n, self._n), y.reshape(1,self._n, self._n))
