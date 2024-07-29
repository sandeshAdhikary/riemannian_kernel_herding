import autograd.numpy as np
from pymanopt.manifolds.special_orthogonal_group import SpecialOrthogonalGroup as Rotations
import torch
from scipy.spatial.transform import Rotation
from src.manifold_utils.ortho_utils import geodesic as geodesic_ortho
from src.manifold_utils.sphere_utils import isSphere

def project(X):
    '''
    Projects a 3x3 matrix X onto orhtoongal matrices with determinant +1
    (The algorithm was designed for 3x3 matrices, so may not work for SO(n) in general)
    :param X: (num_mats, 3, 3)
    :return:
    '''
    
    assert X.shape[1] == X.shape[2] == 3, "Only 3x3 matriecs can be projected onto SO(3)"
    
    # First get the determinant of X
    s = np.linalg.det(X)
    # If determinant is positive, s = 1, else -1
    s = s / np.abs(s)

    
    M = np.transpose(X, (0,2,1)) @ X # M = X^T.X
    D, U = np.linalg.eigh(M)

    D = D[:,::-1] #  Switch order of eigen values to be in lambda1 >= lambda2 >= labmda3
    U = U[:,:,::-1] #  Switch order of eigen values to be in lambda1 >= lambda2 >= labmda3
    D = D**(-0.5)
    D = D * np.array([1., 1., s[0]])

    D = np.apply_along_axis(np.diag, -1, D)

    return (X @ U @ D) @ np.transpose(U, (0,2,1))

def isSO3(X):
    '''
    Check if each matrix in X is orthogonal and that it has determinant == 1
    :param X: (num_samples, dim, dim)
    :return:
    '''
    if len(X.shape) < 3:
        X = np.expand_dims(X, 0)

    ## First check for orhtogonality
    X_XT = np.einsum('ijk,ikn -> ijn', X, np.transpose(X, (0,2,1)))
    eyes = np.expand_dims(np.eye(X.shape[1]), 0)
    eyes = eyes.repeat(X.shape[0], axis=0)
    ortho = np.max(np.abs(X_XT - eyes)) < 1e-8
    
    # Check determinant
    det = np.max(np.linalg.det(X) - np.ones(X.shape[0])) < 1e-8
    
    return (ortho and det)


def geodesic(X, Y):
    '''
    dist(x,y) = ||logm(x^T y) ||
    Compute the above pairwise distances for all matrices x in X and y in Y
    :param X:
    :param Y:
    :return:
    '''
    # TODO: Using the orthogonal geodesic (i.e. the canonical metric for orthogonal matrices for points on
    #       a connected component) since SO(3) is a connected component of O(3)

    X = X if isSO3(X) else project(X)
    Y = Y if isSO3(Y) else project(Y)
    return geodesic_ortho(X, Y)

    # if len(X.shape) < 3:
    #     X = np.expand_dims(X, 0)
    # if len(Y.shape) < 3:
    #     Y = np.expand_dims(Y, 0)
    #
    # X = X if isSO3(X) else project(X)
    # Y = Y if isSO3(Y) else project(Y)
    #
    # dims = X.shape[1]
    # assert dims == X.shape[2], "X must be a tensor of square matrices"
    # assert dims == Y.shape[1] == Y.shape[2], "Dimensions of Y must match X"
    #
    # num_x, num_y = X.shape[0], Y.shape[0]
    #
    # # Compute all pairwise X^T Y
    # prods = (np.transpose(X, (0, 2, 1)) @ Y[:, np.newaxis])
    # prods = prods.reshape(-1, dims, dims)
    #
    # # Get the eigenvectors of the products
    # _, V = np.linalg.eig(prods)
    #
    # # XTY_p = V^{-1} (X^T Y) V (this should be a diagonal matrix)
    # V_inv = np.linalg.inv(V)
    # XTY_p = V_inv @ prods @ V
    #
    #
    # # Get the diagonals of all matrices in Ap, and take scalar logarithm
    # logXTY_p = np.diagonal(XTY_p, axis1=1, axis2=2)  # Get the diagonal elements
    # logXTY_p = np.log(logXTY_p)  # Take log
    # logXTY_p = np.apply_along_axis(np.diag, axis=1, arr=logXTY_p)  # Create diagonal matrices
    #
    # # Get log(X^T Y)
    # logXTY = V @ logXTY_p @ V_inv
    #
    # # Take the norm
    # # dist = np.sqrt(np.add.reduce(((np.real(logXTY) - np.imag(logXTY)) * logXTY).real, axis=(1,2)))
    # logXTYconj = np.real(logXTY) - np.imag(logXTY)
    # dist = np.matmul(logXTYconj, logXTY)
    # dist = np.trace(dist, axis1=1, axis2=2)
    # dist = np.sqrt(dist)
    # dist = np.real(dist)
    # # dist = np.linalg.norm(logXTY, axis=(1, 2))
    # dist = dist.reshape(num_y, num_x).T  # Reshape it into (num_x, num_y)
    #
    # return dist / (np.pi * np.sqrt(dims))

def geodesic_cayley(X,Y,Z):
    '''
    Geodesic distance between points x and y, both on SO(3), based on the Cayley map (retraction)
    at point z, also on SO(3)
    This is computed as the inner product <C_z(x), C_z(y)> where C_z(.) is the Cayley map defined
    at point z on SO(3)
    return: trace(inv_cayley(z,x).T @ inv_cayley(z,y))
    '''
    A = inv_cayley_map(Z, X)
    B = inv_cayley_map(Z, Y)
    return np.diag(A.T @ B).sum()

def cayley_map(X,Z, alpha=1.):
    '''
    Apply the Cayley map C_Z(X)
    '''
    out = np.eye(Z.shape[0]) + alpha*Z
    out = out @ np.linalg.pinv(np.eye(Z.shape[0]) - alpha*Z)
    out = out @ X
    return out


def inv_cayley_map(X, Y):
    '''
    Compute the inv_cayley_map (i.e. cayley lift) [C_X (Y)]^-1.
    See, for example, the map used of averaging on Lie groups/ Stiefel manifolds in
        (1) Empirical Arithmetic Averaging Over theCompact Stiefel Manifold, Kaneko et al (2013
        (2) Particle Filtering on the Stiefel Manifold with Optimal Transport, Wang et al (2020)
    '''

    assert X.shape == Y.shape
    p, n = X.shape

    # First split up x and y
    Xu, Yu = X[0:n, :], Y[0:n, :]
    if n == p:
        Xl, Yl = np.zeros_like(Xu), np.zeros_like(Yu)
    else:
        Xl, Yl = X[n:, :], Y[n:, :]

    # get sum_u_inv since we'll reuse it
    sum_u_inv = np.linalg.pinv(Xu + Yu)

    # Get A
    A = 2*sum_u_inv.T
    A = A @ skew_part( (Yu.T)@Xu + (Xu.T)@Yl)
    A = A @ sum_u_inv

    if n == p:
        return A
    else:
        # Get B
        B = (Yl - Xl) @ sum_u_inv

        # Create the block diagonal output matrix
        C = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[0]))
        C[0:A.shape[0], 0:A.shape[1]] = A
        C[A.shape[0]:, 0:B.shape[1]] = B
        C[0:A.shape[0], A.shape[1]:] = -B.T

        return C


def skew_part(X):
    '''
    Compuate the skew symmetric portion of a matrix
    '''
    return 0.5*(X.T - X)

def centroid(X, weights, num_iters, T_init = None):
    '''
    :param X: array of shape (n,dims). Compute the Riemannian centroid of the n samples in X
    :num_iters: Number of iterations to run
    :T_init: Initial guess for the centroid. If None, use simple average
    '''

    assert X.shape[0] == weights.shape[0], "Num samples and num weights must be the same"

    # T = T_init if T_init is not None else np.mean(X, axis=0)
    T = T_init if T_init is not None else X[np.random.choice(range(X.shape[0])), :]
    for _ in np.arange(num_iters):

        # Loop through the samples and add up the Cayley lifts
        iter_sum = np.zeros_like(T)
        for ids in np.arange(X.shape[1]):
            iter_sum += inv_cayley_map(T, X[ids, :]) * weights[ids]

        #  Project onto manifold with Cayley map
        T = cayley_map(T, iter_sum)

    return T

def rotmat_to_quats(rotmat):
    # Convert a rotation matrix (SO3) into quaternions
    assert isSO3(rotmat), "The rotation matrix is not on SO3"
    Q = []
    for batch in np.arange(rotmat.shape[0]):
        q = Rotation.from_matrix(rotmat[batch,:,:]).as_quat()
        Q.append(q)
    Q = np.stack(Q)

    if torch.is_tensor(rotmat):
        Q = torch.from_numpy(Q)

    return Q

def quats_to_rotmat(quats):
    '''
    Convert a 4 dimensional quaternion into a 3x3 rotation matrix on SO(3)
    '''

    assert isSphere(quats)
    R = []
    for batch in np.arange(quats.shape[0]):
        r = Rotation.from_quat(quats[batch,:]).as_matrix()
        R.append(r)
    R = np.stack(R)
    if torch.is_tensor(quats):
        R = torch.from_numpy(R)
    return R



if __name__ == "__main__":
    # Check centroid function
    from scipy.stats import special_ortho_group
    dims = 3
    num_data = 10
    X = data = special_ortho_group.rvs(dims, size=num_data)
    weights = np.ones(X.shape[0])
    num_iters = 10
    X_centroid = centroid(X, weights, num_iters)

