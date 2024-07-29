import numpy as np
from pymanopt.manifolds.special_orthogonal_group import SpecialOrthogonalGroup
from src.kernels.kernels import RotationKernel

man = SpecialOrthogonalGroup(3)

num_samples = 10
x = np.zeros((num_samples, 3, 3))
y = np.zeros((num_samples, 3, 3))
for idx in range(num_samples):
    x[idx, :, :] = man.rand()
    y[idx, :, :] = man.rand()

kernel = RotationKernel(type=2)
out = kernel.kernel_eval(x,y)
