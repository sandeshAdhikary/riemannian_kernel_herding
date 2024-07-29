import autograd.numpy as np
import torch
from src.manifold_utils.euclidean_utils import geodesic as euclidean_dist
import geoopt
from pymanopt.manifolds import SymmetricPositiveDefinite
torch.set_default_dtype(torch.double)

EIG_TOLERANCE = 1e-8
geoopt_spd_manifold = geoopt.SymmetricPositiveDefinite()

def geodesic(X, Y, backend="geoopt"):
    '''
    Geodesic distance on SPD manifold
    :param X: (num_samples, dims, dims)
    :param Y: (num_samples, dims, dims)
    :return:
    '''

    X_ = X if isPD(X) else project(X)
    Y_ = Y if isPD(Y) else project(Y)


    if backend == "geoopt":

        if not torch.is_tensor(X_):
            X_ = torch.from_numpy(X_)
        if not torch.is_tensor(Y_):
            Y_ = torch.from_numpy(Y_)

        if X_.dim() < Y_.dim():
            # TODO: X_ should have shape (1,..). Will this check always work?
            X_ = X_.unsqueeze(0)
        # TODO: Do this without list comprehension
        out = torch.stack([geoopt_spd_manifold._affine_invariant_metric(X_[idx, :], Y_) for idx in np.arange(X_.shape[0])])
    else:
        raise(NotImplementedError)
        ## DEBUGGING: Euclidean distance for debugging
        #
        # out = euclidean_dist(X_,Y_)
        # return out


        ## The affine-invariant distance
        # X_ = X if isPD(X) else project(X)
        # Y_ = Y if isPD(Y) else project(Y)

        # dims = X.shape[1]
        # if X.shape[0] == 1:
        # # First project the matrices
        # X_ = X if isPD(X) else project(X)
        # Y_ = Y if isPD(Y) else project(Y)
        #
        # c = np.linalg.cholesky(X_)
        # c_inv = np.linalg.inv(c)
        # logm = multilog(multiprod(multiprod(c_inv, Y_), multitransp(c_inv)),
        #                 pos_def=True)
        #
        # out = np.linalg.norm(logm, axis=(1,2)).reshape(1,-1)
        # else:
        #     raise(NotImplementedError)
        #
        # return out

        ## The Log-Frobenius/Log-Euclidean distance
        # First compute the matrix logs
        # X_ = X if isPD(X) else project(X)
        # Y_ = Y if isPD(Y) else project(Y)
        # log_X = multilog(X_, pos_def=True)
        # log_Y = multilog(Y_, pos_def=True)
        # #
        # # # Vectorize them
        # log_X = log_X.reshape(log_X.shape[0], -1)
        # log_Y = log_Y.reshape(log_Y.shape[0], -1)
        # #
        # #
        # # # Compute Euclidean (Frobenius) distance between the vectors
        # out = euclidean_dist(log_X, log_Y)
    #
    if torch.is_tensor(out):
        if torch.any(torch.isnan(out)):
            raise (ValueError("NaNs found in distance matrix!"))
    else:
        if np.any(np.isnan(out)):
            raise(ValueError("NaNs found in distance matrix!"))

    return out
    # return out/np.sqrt(dims)


def project(X, backend="geoopt"):
    """
    Project all matrices in X onto SPD
    :param X: [num_matrices, dim, dim]
    :return:
    """
    if isPD(X):
        return X
    else:
        if torch.is_tensor(X):
            X_proj = geoopt_spd_manifold.projx(X)
            # if torch.norm(X_proj - X_proj.permute(0, 2, 1)) > 1e-10:
            #     for idx in np.arange(X_proj.shape[0]):
            #         if torch.norm(X_proj[idx,:] - X_proj[idx,:].T) > 1e-10:
            #             X_proj[idx,:] = X_proj[idx,:] + (X_proj[idx,:].T)
            #             X_proj[idx, :] = X_proj[idx,:]*0.5
            #     X_proj = geoopt_spd_manifold.projx(X_proj)
        else:
            X_proj = geoopt_spd_manifold.projx(torch.from_numpy(X))
            # if np.linalg.norm(X_proj - np.transpose(X_proj, (0, 2, 1))) > 1e-10:
            #     for idx in np.arange(X_proj.shape[0]):
            #         if np.linalg.norm(X_proj - np.transpose(X_proj)):
            #             X_proj[idx, :] = X_proj[idx, :] + np.transpose(X_proj[idx, :])
            #             X_proj[idx, :] = X_proj[idx, :] * 0.5
            #     X_proj = geoopt_spd_manifold.projx(X_proj)

        assert isPD(X_proj), "Not SPD even after projection"
        return X_proj


def project_matrix(A):
    """
    Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """


    ## Code from : https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd

    # return np.matmul(A, A.T)

    B = (A + A.T) / 2

    if torch.is_tensor(B):
        # Torch
        _, s, V = torch.svd(B, )
        H = V.matmul(torch.diag(s)).matmul(V.T)
        A2 = (B+H)/2
        A3 = (A2 + A2.T)/2
    else:
        # Numpy
        _, s, V = np.linalg.svd(B, full_matrices = False)
        # H = V*Sigma*V.T ; Here, order is reversed since np.linglg.svd retursn U,S,V.T
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

    if torch.is_tensor(A3):
        assert torch.max(torch.abs(A3 - A3.t())) < 1e-10, "Projected matrix is not symmetric"
    else:
        assert np.max(np.abs(A3 - A3.T)) < 1e-10, "Projected matrix is not symmetric"

    if isPD(A3):
        return A3
    # spacing = np.spacing(np.linalg.norm(A))
    spacing = 1e-16
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    if torch.is_tensor(A3):
        I = torch.eye(A3.shape[0])
        k = 1
        while not isPD(A3):
            #TODO: Should replace this with eigsym?
            mineig = torch.min(torch.real(torch.linalg.symeig(A3)[0]))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

            if k > 10:
                # Just can't seem to project. In this case, let's just do an extreme projection (i.e. not the nearest SPD)
                # Set eigenvalues to be absolute values (so they're real and positive) and add some noise to prevent 0
                eigvals, eigvecs = np.linalg.symeig(A3)
                A3 = eigvecs.matmul(torch.diag(torch.abs(eigvals) + 1e-10)).matmul(torch.pinverse(eigvecs))
                assert isPD(A3), "Matrix is not PD even after extreme projection"
    else:
        I = np.eye(A3.shape[0])
        k = 1
        while not isPD(A3):
            # print("Adding noise...")
            mineig = np.min(np.real(np.linalg.eigh(A3)[0]))

            A3 += I * (-mineig * k**2 + spacing)
            k += 1

            if k > 10:
                # Just can't seem to project. In this case, let's just do an extreme projection (i.e. not the nearest SPD)
                # Set eigenvalues to be absolute values (so they're real and positive) and add some noise to prevent 0
                eigvals, eigvecs = np.linalg.eigh(A3)
                A3 = eigvecs @ np.diag(np.abs(eigvals) + 1e-10) @ np.linalg.pinv(eigvecs)


    # #Attempt 1: Add some noise until it is PD
    # noise = 10e-16
    # I = torch.eye(A3.shape[0]) if torch.is_tensor(A3) else np.eye(A3.shape[0])
    # k = 0
    # while not isPD(A3):
    #     A3 += I * (noise)
    #     noise = noise*10
    #     k += 1
    #     if k > 15:
    #         # Extreme projection: Just flip the negative eigenvalues
    #         raise(Exception("Could not make matrix PSD, even with regularization noise ~1"))
    #
    return A3


def apply_along_batches(func, M):
    """
    Apply function func to all matrices in M
    M is assumed to be of shape [num_batches, dim1, dim2]
    :param func:
    :param M:
    :return:
    """

    if torch.is_tensor(M):
        tList = [func(m) for m in torch.unbind(M, dim=0)]
        res = torch.stack(tList, dim=0)
    else:
        tList = [func(M[idx, :, :]) for idx in np.arange(M.shape[0])]
        res = np.stack(tList, 0)


    return res

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    ## Code from : https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd


    try:
        # If cholesky fails, it must not be SPD
        if torch.is_tensor(B):
            _ = torch.cholesky(B)
        else:
            _ = np.linalg.cholesky(B)

        # Make sure it is symmetric too
        if torch.is_tensor(B):
            isSym = torch.norm(B - B.permute((0, 2, 1))) < 1e-10
        else:
            isSym = np.linalg.norm(B - np.transpose(B, (0, 2, 1))) < 1e-10

        if isSym:
            return True
        else:
            return False
    except:
        # Cholesky failed so return False
        return False

def multiprod(A, B):
    """
    Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    assumed to be arrays containing M matrices, that is, A and B have
    dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    in A with the corresponding matrix in B, using matrix multiplication.
    so multiprod(A, B) has dimensions (M, N, Q).
    """

    # Code from: pymanopt.tools.multi

    if not torch.is_tensor(A) and not torch.is_tensor(B):
        # First check if we have been given just one matrix
        if A.ndim == 2:
            return np.dot(A, B)

        # Approx 5x faster, only supported by numpy version >= 1.6:
        return np.einsum('ijk,ikl->ijl', A, B)
    elif torch.is_tensor(A) and torch.is_tensor(B):
        # First check if we have been given just one matrix
        if A.ndim == 2:
            return torch.dot(A, B)

        # Approx 5x faster, only supported by numpy version >= 1.6:
        return torch.einsum('ijk,ikl->ijl', A, B)
    else:
        raise (ValueError('Both tensors must be of same type (np or torch tensors)'))


def multilog(A, pos_def=False):
    # Code from: pymanopt.tools.multi


    if not pos_def:
        raise NotImplementedError

    # Computes the logm of each matrix in an array containing k positive
    # definite matrices. This is much faster than scipy.linalg.logm even
    # for a single matrix. Could potentially be improved further.
    if not torch.is_tensor(A):
        w, v = np.linalg.eigh(A)
        # TODO: Setting 0 eigenvalues to be slightly non-zero
        # But we shouldn't have zero eigenvalues if pos_def
        if np.any(w <= 0):
            non_pos = w[w <= 0]
            if np.all(np.abs(non_pos) < EIG_TOLERANCE):
                # Negative/zero within tolerance, so just replace with small +noise
                noise = np.zeros_like(w)
                noise[w <= 0] = EIG_TOLERANCE
                w = w + noise
            else:
                raise(ValueError("Negative eigenvalue! A is not symm. pos. def"))

        w = np.expand_dims(np.log(w), axis=-1)
    else:
        w, v = torch.symeig(A, eigenvectors=True)
        if torch.any(w==0):
            non_pos = w[w <= 0]
            if np.all(torch.abs(non_pos) < 1e-8):
                # Negative/zero within tolerance, so just replace with small +noise
                noise = torch.zeros_like(w)
                noise[w <= 0] = 1e-8
                w = w + noise
            else:
                raise (ValueError("Negative eigenvalue! A is not symm. pos. def"))
        w = torch.log(w).unsqueeze(-1)

    return multiprod(v, w * multitransp(v))


def multitransp(A):
    """
    Inspired by MATLAB multitransp function by Paolo de Leva. A is assumed to
    be an array containing M matrices, each of which has dimension N x P.
    That is, A is an M x N x P array. Multitransp then returns an array
    containing the M matrix transposes of the matrices in A, each of which
    will be P x N.
    """
    # First check if we have been given just one matrix
    if A.ndim == 2:
        return A.T

    if torch.is_tensor(A):
        return A.permute((0, 2, 1))
    else:
        return np.transpose(A, (0, 2, 1))


def constraint_violations(X):
    """
    Get constraint violation errors for all matrices in X
    Error: 0 if smallest eigenvalue is > 0; else abs(smallest_eigenvalue)
    :param X:
    :return:
    """
    if torch.is_tensor(X):
        out = torch.min(torch.eig(X)[0], dim=1)
        out = torch.abs(out * (out < 0))
    else:
        out = np.min(np.linalg.eig(X)[0], axis=1)
        out = np.abs(out*(out < 0))

    return out


def num_lower_triu_elems(n):
    '''
    Given a matrix dimenion, compute the number of lower triangular elements
    e.g. for a 3x3 matrix n=3
    '''

    return int((n * (n - 1)) / 2) + n

def vec_to_chol(vec, dims):
    '''
    Convert a vector into a lower triangular matrix i.e. a Cholesky component
    '''
    xc = torch.cat([vec[dims:], vec.flip(dims=[0])])
    y = xc.view(dims, dims)
    return torch.tril(y)

def chol_to_spd(chol):
    '''
    Convert a cholesky factor into a full SPD matrix
    '''
    return chol.matmul(chol.T)



if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)

    # Generate a bunch of random 2x2 matrices
    init_mats = np.random.random((10, 2, 2))

    constr_viols = constraint_violations(init_mats)
    print("Initial constraint violations: {}".format(constr_viols))
    projected_mats = project(init_mats)

    constr_viols = constraint_violations(projected_mats)
    print("Projected constraint violations: {}".format(constr_viols))







