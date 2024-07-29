import geoopt
from geoopt.manifolds.product import ProductManifold
import torch
from typing import Tuple
from src.manifold_utils.product import get_submanifold_tensors

class PoseManifold(ProductManifold):
    '''
    Pose manifold constructed as a product space of R(pos_dim) and S(orientation_dim-1)
    This manifold is not equivalent to the special euclidean group
    '''
    def __init__(self, pos_dim, orientation_dim):
        self.pos_dim, self.orientation_dim = pos_dim, orientation_dim
        self.manifold_backend = 'geoopt'
        super().__init__(
                    (geoopt.Euclidean(), pos_dim),
                    (geoopt.Sphere(), orientation_dim)
                )

    def sub_tensor_fn(self, x, idx):

        assert idx < len(self.manifolds), "The product manifold has less than {} sub-manifolds".format(idx + 1)

        if torch.is_tensor(x):
            return self.take_submanifold_value(x, idx)
        else:
            return self.take_submanifold_value(torch.from_numpy(x), idx)

class EuclideanPoseManifold(ProductManifold):
    '''
    Pose manifold constructed as a product space of R(pos_dim) and R(orientation_dim)
    This manifold is not equivalent to the special euclidean group
    '''
    def __init__(self, pos_dim, orientation_dim):
        self.pos_dim, self.orientation_dim = pos_dim, orientation_dim
        self.manifold_backend = 'geoopt'
        super().__init__(
                    (geoopt.Euclidean(), pos_dim),
                    (geoopt.Euclidean(), orientation_dim)
                )

    def sub_tensor_fn(self, x, idx):

        assert idx < len(self.manifolds), "The product manifold has less than {} sub-manifolds".format(idx + 1)

        if torch.is_tensor(x):
            return self.take_submanifold_value(x, idx)
        else:
            return self.take_submanifold_value(torch.from_numpy(x), idx)

class MultiAgentPoseManifold(ProductManifold):
    '''
    Multi-agent pose manifold: Direct sum of as many pose manifolds as there are agents
    '''
    def __init__(self, num_agents, pose_manifold):
        self.num_agents = num_agents
        self.manifold_backend = 'geoopt'
        multi_agent_tuple = tuple((pose_manifold, 4) for _ in range(self.num_agents))
        super().__init__(
            *multi_agent_tuple
        )


    def sub_tensor_fn(self, x, idx):

        assert idx < self.num_agents, "The product manifold has less than {} sub-manifolds".format(idx + 1)

        if torch.is_tensor(x):
            return self.take_submanifold_value(x, idx)
        else:
            return self.take_submanifold_value(torch.from_numpy(x), idx)

if __name__ == "__main__":
    pass