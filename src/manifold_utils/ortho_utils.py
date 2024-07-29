import autograd.numpy as np
import torch
from geoopt.manifolds import Stiefel

stiefel_manifold = Stiefel()

def project(X):
    '''
    Project X to orthonormal matrices
    Note: This projection does not ensure the determinant will be +1. So, this is not a
    projection onto Rotations, but the closest we can do
    :param X:
    :return:
    '''
    if torch.is_tensor(X):
        raise(NotImplementedError)
    else:
        U, _, Vh = np.linalg.svd(X, full_matrices=False)
        Sigma = np.expand_dims(np.eye(X.shape[1]), 0).repeat(X.shape[0], axis=0)
        return np.einsum('ijk, ikl, ilm -> ijm', U, Sigma, Vh)


def isOrtho(X):
    '''
    Check if each matrix in X is orthogonal.
    :param X: (num_samples, dim, dim)
    :return:
    '''
    if torch.is_tensor(X):
        X_XT = torch.einsum('ijk,ikn -> ijn', X, X.permute(0, 2, 1))
        eyes = torch.eye(X.shape[1]).unsqueeze(0)
        eyes = torch.repeat_interleave(eyes, X.shape[0], 0)
        return torch.max(torch.abs(X_XT - eyes)) < 1e-10
    else:
        if len(X.shape) < 3:
            X = np.expand_dims(X,0)

        X_XT = np.einsum('ijk,ikn -> ijn', X, np.transpose(X, (0, 2, 1)))
        eyes = np.expand_dims(np.eye(X.shape[1]), 0)
        eyes = eyes.repeat(X.shape[0], axis=0)
        return np.max(np.abs(X_XT - eyes)) < 1e-10


def geodesic(X, Y):
    '''
    Equation 2.15 in "The Geometry of Algorithms with Orthogonality Constraints"
                        Edelmen et al
    Compute the above pairwise distances for all matrices x in X and y in Y
    :param X:
    :param Y:
    :return:
    '''

    if len(X.shape) < 3:
        X = np.expand_dims(X, 0)
    if len(Y.shape) < 3:
        Y = np.expand_dims(Y, 0)

    X = X if isOrtho(X) else project(X)
    Y = Y if isOrtho(Y) else project(Y)

    dims = X.shape[1]
    assert dims == X.shape[2], "X must be a tensor of square matrices"
    assert dims == Y.shape[1] == Y.shape[2], "Dimensions of Y must match X"

    if torch.is_tensor(X) and torch.is_tensor(Y):
        # Compute X^T Y
        XTY = torch.einsum('ijk, lkn -> iljn', X.permute(0, 2, 1), Y)
        XTY = torch.squeeze(XTY)

        # Get eigenvalues of XTY
        eigvals = torch.eig(XTY)[0]

        # Solve for thetas
        thetas = 1j * torch.log(eigvals)  # 1j = sqrt(-1)

        # Distance is the two norm of thetas
        # TODO: The Edelman paper says sqrt(\sum_i theta_i**2)
        dists = torch.linalg.norm(thetas, axis=2)

        return dists / torch.sqrt(dims)

    elif not torch.is_tensor(X) and not torch.is_tensor(Y):
        # Compute X^T Y
        XTY = np.einsum('ijk, lkn -> iljn', np.transpose(X, (0, 2, 1)), Y)

        # Get eigenvalues of XTY
        eigvals = np.linalg.eig(XTY)[0]

        # Solve for thetas
        thetas = 1j*np.log(eigvals) # 1j = sqrt(-1)

        # Distance is the two norm of thetas
        # TODO: The Edelman paper says sqrt(\sum_i theta_i**2)
        dists = np.linalg.norm(thetas, axis=2)

        return dists/np.sqrt(dims)
    else:
        raise(ValueError("One array is a torch tensor, and the other is not"))

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



if __name__ == "__main__":
    # Check centroid function
    from scipy.stats import ortho_group
    dims = 3
    num_data = 10
    X = data = ortho_group.rvs(dims, size=num_data)
    weights = np.ones(X.shape[0])
    num_iters = 10
    X_centroid = centroid(X, weights, num_iters)
    print("Done")