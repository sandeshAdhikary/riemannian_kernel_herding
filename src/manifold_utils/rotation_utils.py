import autograd.numpy as np
from pymanopt.manifolds.special_orthogonal_group import SpecialOrthogonalGroup as Rotations
import scipy

def project(X):
    '''
    Project X to orthonormal matrices
    Note: This projection does not ensure the determinant will be +1. So, this is not a
    projection onto Rotations, but the closest we can do
    :param X:
    :return:
    '''
    U,_,Vh = np.linalg.svd(X)
    Sigma = np.expand_dims(np.eye(X.shape[1]), 0).repeat(X.shape[0], axis=0)
    return np.einsum('ijk, ikl, ilm -> ijm', U, Sigma, Vh)


def project_det(X):
    '''
    Projects a 3x3 matrix X onto orhtoongal matrices with determinant +1
    (The algorithm was designed for 3x3 matrices, so may not work for SO(n) in general)
    :param X: (num_mats, 3, 3)
    :return:
    '''
    
    assert X.shape[1] == X.shape[2] == 3, "Can only project 3x3 matriecs onto SO(3)"
    
    # First get the determinant of X
    s = np.linalg.det(X)
    # If determinant is positive, s = 1, else -1
    s = s / np.abs(s)

    
    M = np.transpose(X, (0,2,1)) @ X # M = X^T.X
    D, U = np.linalg.eigh(M)
    # U = np.transpose(U, (0,2,1))

    # D = np.flip(D, axis=1) # Switch order of eigen values to be in lambda1 >= lambda2 >= labmda3
    # U = np.flip(U, axis=2)
    D = D[:,::-1]
    U = U[:,:,::-1]
    D = D**(-0.5)
    # S = np.ones_like(D)
    # S[:, 2] = s
    D = D * np.array([1.,1.,s[0]])

    D = np.apply_along_axis(np.diag, -1, D)

    return (X @ U @ D) @ np.transpose(U, (0,2,1))



def isOrtho(X):
    '''
    Check if each matrix in X is orthogonal.
    :param X: (num_samples, dim, dim) 
    :return: 
    '''

    X_XT = np.einsum('ijk,ikn -> ijn', X, np.transpose(X, (0,2,1)))
    eyes = np.expand_dims(np.eye(X.shape[1]), 0)
    eyes = eyes.repeat(X.shape[0], axis=0)
    return np.max(np.abs(X_XT - eyes)) < 1e-10

def geodesic(X, Y):
    '''
    dist(x,y) = ||logm(x^T y) ||
    Compute the above pairwise distances for all matrices x in X and y in Y
    :param X:
    :param Y:
    :return:
    '''
    
    X = X if isOrtho(X) else project(X)
    Y = Y if isOrtho(Y) else project(Y)

    dims = X.shape[1]
    assert dims == X.shape[2], "X must be a tensor of square matrices"
    assert dims == Y.shape[1] == Y.shape[2], "Dimensions of Y must match X"

    num_x, num_y = X.shape[0], Y.shape[0]

    # Compute all pairwise X^T Y
    prods = (np.transpose(X, (0, 2, 1))@Y[:, np.newaxis])
    prods = prods.reshape(-1, dims, dims)

    # Get the eigenvectors of the products
    _,V = np.linalg.eig(prods)


    # XTY_p = V^{-1} (X^T Y) V (this should be a diagonal matrix)
    V_inv = np.linalg.inv(V)
    XTY_p =  V_inv @ prods @ V

    # TODO: Check to make sure this is a diagonal matrix

    # Get the diagonals of all matrices in Ap, and take scalar logarithm
    logXTY_p = np.diagonal(XTY_p, axis1=1, axis2=2) # Get the diagonal elements
    logXTY_p = np.log(logXTY_p) # Take log
    logXTY_p = np.apply_along_axis(np.diag, axis=1, arr=logXTY_p) # Create diagonal matrices

    # Get log(X^T Y)
    logXTY = V @ logXTY_p @ V_inv

    # Take the norm
    dist = np.linalg.norm(logXTY, axis=(1, 2))
    dist = dist.reshape(num_y, num_x).T # Reshape it into (num_x, num_y)

    return dist/(np.pi * np.sqrt(dims))

def isRot(X):
    '''
    Check if each matrix in X is orthogonal and that it has determinant == 1
    :param X: (num_samples, dim, dim)
    :return:
    '''
    
    ## First check for orhtogonality
    X_XT = np.einsum('ijk,ikn -> ijn', X, np.transpose(X, (0,2,1)))
    eyes = np.expand_dims(np.eye(X.shape[1]), 0)
    eyes = eyes.repeat(X.shape[0], axis=0)
    ortho = np.max(np.abs(X_XT - eyes)) < 1e-8
    
    # Check determinant
    det = np.max(np.linalg.det(X) - np.ones(X.shape[0])) < 1e-8
    
    return (ortho and det)


def geodesic_det(X, Y):
    '''
    dist(x,y) = ||logm(x^T y) ||
    Compute the above pairwise distances for all matrices x in X and y in Y
    :param X:
    :param Y:
    :return:
    '''

    X = X if isRot(X) else project_det(X)
    Y = Y if isRot(Y) else project_det(Y)

    dims = X.shape[1]
    assert dims == X.shape[2], "X must be a tensor of square matrices"
    assert dims == Y.shape[1] == Y.shape[2], "Dimensions of Y must match X"

    num_x, num_y = X.shape[0], Y.shape[0]

    # Compute all pairwise X^T Y
    prods = (np.transpose(X, (0, 2, 1)) @ Y[:, np.newaxis])
    prods = prods.reshape(-1, dims, dims)

    # Get the eigenvectors of the products
    _, V = np.linalg.eig(prods)

    # XTY_p = V^{-1} (X^T Y) V (this should be a diagonal matrix)
    V_inv = np.linalg.inv(V)
    XTY_p = V_inv @ prods @ V

    # TODO: Check to make sure this is a diagonal matrix

    # Get the diagonals of all matrices in Ap, and take scalar logarithm
    logXTY_p = np.diagonal(XTY_p, axis1=1, axis2=2)  # Get the diagonal elements
    logXTY_p = np.log(logXTY_p)  # Take log
    logXTY_p = np.apply_along_axis(np.diag, axis=1, arr=logXTY_p)  # Create diagonal matrices

    # Get log(X^T Y)
    logXTY = V @ logXTY_p @ V_inv

    # Take the norm
    dist = np.linalg.norm(logXTY, axis=(1, 2))
    dist = dist.reshape(num_y, num_x).T  # Reshape it into (num_x, num_y)

    return dist / (np.pi * np.sqrt(dims))


if __name__ == "__main__":
    
    
    # Check to see if projection works as intended
    num_samples = 10
    num_trials = 20
    for idx in np.arange(num_trials):
        print("Running {} of {}".format(idx, num_trials))
        X = np.random.random((num_samples,3,3))
        Xp = project_det(X)

        eyes =  np.eye(3)[np.newaxis,:].repeat(num_samples, axis=0)
        left_ortho_err = np.linalg.norm((np.transpose(Xp, (0, 2, 1)) @ Xp) - eyes)
        right_ortho_err = np.linalg.norm(Xp @ (np.transpose(Xp, (0, 2, 1))) - eyes)
        det_err = np.linalg.norm(np.linalg.det(Xp) - np.ones(num_samples))
        assert left_ortho_err < 1e-8, "Left ortho err: {}".format(left_ortho_err)
        assert right_ortho_err < 1e-8, "Right ortho err: {}".format(right_ortho_err)
        assert det_err < 1e-8, "Determinant err: {}".format(det_err)

    print("Done")