
from abc import ABC, abstractmethod
import torch
import autograd.numpy as np
# import numpy as np
from src.manifold_utils.euclidean_utils import geodesic as euclid_dist
from src.manifold_utils.spd_utils import geodesic as spd_dist
from src.manifold_utils.sphere_utils import geodesic as sphere_dist
from src.manifold_utils.sym_utils import geodesic as sym_dist
from src.manifold_utils.rotation_utils import geodesic as rot_dist
from src.manifold_utils.rotation_utils import geodesic_det as rot_det_dist
from src.manifold_utils.oblique_utils import geodesic as oblique_dist
from src.manifold_utils.ortho_utils import geodesic as ortho_dist
from src.manifold_utils.so3_utils import geodesic as so3_dist
from src.manifold_utils.so3_utils import isSO3

class BaseKernel(ABC):
    ### Adapted from code from Sasha Lambert

    def __init__(self):
        self.bandwidth = None

    @abstractmethod
    def kernel_eval(self, x, Y):
        """
        Evaluate kernel
        """
        pass

    def get_pairwise_distances(self, X, Y, manifold=None):

        if manifold is None or manifold == "Euclidean":
            dist_func = euclid_dist
        elif manifold == "Sphere":
            dist_func = sphere_dist
        elif manifold == "SPD":
            dist_func = spd_dist
        elif manifold == "Symmetric":
            dist_func = sym_dist
        elif manifold == "Rotation":
            dist_func = rot_dist
        elif manifold == "RotationDet":
            # TODO: Merge this with Rotation later
            dist_func = rot_det_dist
        elif manifold == "SO3":
            dist_func = so3_dist
        elif manifold == "Oblique":
            dist_func = oblique_dist
        elif manifold == "Orthogonal":
            dist_func = ortho_dist
        else:
            raise(Exception("Unknown manifold"))

        return dist_func(X, Y)


class RBF(BaseKernel):
    """
        # RBF kernel:  k(x, x') = exp( bandwidth*dist(x - x')^2)
        where dist is either euclidean distance, or geodesic distance on a manifold
    """
    def __init__(
            self,
            bandwidth=1.,
            manifold=None,
    ):
        self.bandwidth = bandwidth
        self.manifold = manifold
        self.kernel_type = "RBF"

    def kernel_eval(self, X, Y):
        """
        """

        K = self.get_pairwise_distances(X, Y, self.manifold)
        K = K**2 # Use squared distance
        K *= -self.bandwidth
        if torch.is_tensor(K):
            K = torch.exp(K)
            assert not torch.any((torch.isnan(K))), "Kernel matrix has NaNs!"
            assert K.dtype == torch.double, "Kernel matrix must be double precision"
        else:
            K = np.exp(K)
            assert not np.any((np.isnan(K))), "Kernel matrix has NaNs!"
            assert K.dtype == np.double, "Kernel matrix must be double precision"

        assert sum(sum(K < 0)) == 0, "Kernel matrix has negative entries!"


        return K

    def update_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth


class Laplacian(BaseKernel):
    """
        Laplacian kernel:  k(x, x') = exp( bandwidth*dist(x - x'))
        where dist is either euclidean distance, or geodesic distance on a manifold
    """
    def __init__(
            self,
            bandwidth=1.,
            manifold=None,
            **kwargs,
    ):
        # self.ell = bandwidth
        self.bandwidth = bandwidth
        self.manifold = manifold
        self.kernel_type = "Laplacian"

    def kernel_eval(self, X, Y):
        """
        """

        K = self.get_pairwise_distances(X, Y, self.manifold)
        K *= -self.bandwidth
        if torch.is_tensor(K):
            K = torch.exp(K)
            assert not torch.any((torch.isnan(K))), "Kernel matrix has NaNs!"
            assert K.dtype == torch.double, "Kernel matrix must be double precision"
        else:
            K = np.exp(K)
            assert not np.any((np.isnan(K))), "Kernel matrix has NaNs!"
            assert K.dtype == np.double, "Kernel matrix must be double precision"

        assert sum(sum(K < 0)) == 0, "Kernel matrix has negative entries!"


        return K

    def update_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth

class SumKernel():
    def __init__(self, kernels, sub_tensors_fn):
        '''
        kernels: A list of kernels forming the sum kernel
        sub_tensors_fn: A function to extract tensors for individual kernels
                            input: (point,idx);
                            output: portion of point in the idx-th submanifold
        '''
        self.num_kernels = len(kernels)
        self.kernels = kernels
        self.sub_tensors_fn = sub_tensors_fn
        self.bandwidths = []
        for kernel in self.kernels:
            if isinstance(kernel, type(self)):
                # The kernel is also a product kernel
                self.bandwidths.append(kernel.bandwidths)
            else:
                self.bandwidths.append(kernel.bandwidth)

    def kernel_eval(self, X, Y):
        out = 0
        for idx in range(self.num_kernels):
            x_idx = self.sub_tensors_fn(X, idx)
            y_idx = self.sub_tensors_fn(Y, idx)
            out += self.kernels[idx].kernel_eval(x_idx, y_idx)
        return out

    def update_bandwidth(self, bandwidths):

        assert len(bandwidths) == self.num_kernels, "Num bandwidths not equal to num kernels"

        for idx in range(self.num_kernels):
            self.kernels[idx].update_bandwidth(bandwidths[idx])
            self.bandwidths[idx] = bandwidths[idx]

class RotationKernel():
    """
    Characteristic kernels for the SO(3) group as defined in Eq 7 (type=1) and 8 (type=2) of
    "Characteristic Kernels on Structured Domains Excel in Robotics and Human Action Recognition"

    """

    def __init__(self, type, bandwidth=None):
        assert type in [1, 2], "Unknown type for RotationKernel. Pick from [1,2]"
        self.type = type
        self.bandwidth = bandwidth if type == 2 else None


    def kernel_eval(self, x, y):

        assert isSO3(x) and isSO3(y), "Either x or y are not valid SO(3) matrices"

        theta = self.get_theta(x, y)
        sin_theta = torch.sin(theta) if torch.is_tensor(theta) else np.sin(theta)

        if self.type == 1:
            out = theta * (np.pi - theta)/(8*sin_theta)
        elif self.type == 2:
            out = 2*self.bandwidth*sin_theta/(1 - self.bandwidth**2)
            if torch.is_tensor(out):
                out = torch.atan2(torch.sin(out), torch.cos(out)) / (2*sin_theta)
            else:
                out = np.arctan2(np.sin(out), np.cos(out))/ (2*sin_theta)
        else:
            raise(ValueError("Unknown type for RotationKernel. Pick from [1,2]"))

        if torch.is_tensor(x):
            assert not torch.any(torch.isnan(out))
        else:
            assert not np.any(np.isnan(out))

        return out

    def get_theta(self, x, y):
        '''
        Compute theta = arccos(0.5 * Tr(y^-1 x))
        Note1: For SO(3), y^-1 = y^T
        Note2: We require 0 <= theta <= pi
        '''

        if torch.is_tensor(x):
            assert not torch.any(torch.isnan(x))
            assert not torch.any(torch.isnan(y))
        else:
            assert not np.any(np.isnan(x))
            assert not np.any(np.isnan(y))

        x_ = x if len(x.shape) == 3 else x.reshape(1, x.shape[0], x.shape[1])
        y_ = y if len(y.shape) == 3 else y.reshape(1, y.shape[0], y.shape[1])

        # TODO: The formula for theta in the reference can result in invalid values for cos(theta)
        #       Are they missing a -1? We've added the -1 in this implementation
        if torch.is_tensor(x_) and torch.is_tensor(y_):
            raise(NotImplementedError)
            # theta = torch.einsum('bnm, bnq -> bmq', y_, x_) # Transpose y and then matmul across batches
            # theta = 0.5*(torch.einsum('bmm', theta) - 1) # Trace across batches
            # theta = torch.arccos(theta)
            # assert torch.all(theta <= np.pi) and torch.all(theta >= 0)
        elif ~torch.is_tensor(x_) and ~torch.is_tensor(y_):
            # pairwise products y^T . x
            theta = np.einsum('bij, Bjq -> bBiq', y_.transpose(0, 2, 1), x_)
            # Compute trace: Tr(y^T . x)
            theta = theta.reshape(y_.shape[0]*x_.shape[0], 3, 3)
            theta = theta.diagonal(offset=0, axis1=-1, axis2=-2).sum(-1) # Trace: roundabout since autograd cant handle trace directly
            theta = theta.reshape((y_.shape[0], x_.shape[0]))

            # cos(theta) = (Tr(y^T . x) - 1) * 0.5
            theta = (theta)*0.5
            if np.any(np.abs(theta) > 1):
                assert np.max(np.abs(theta)) - 1 <= 1e-8 # Error to correct is not too egregious
                theta = np.clip(theta, -(1-1e-8), 1-1e-8)
            if np.any(np.isnan(theta)):
                raise(ValueError, "NANs found in arccos(theta)")
            theta = np.arccos(theta)
            if np.any(np.isnan(theta)):
                raise (ValueError, "NANs found in theta")
        else:
            raise(ValueError("One of x and y is a torch tensor and the other is not. They should be of the same type"))

        return theta.T

    def update_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth