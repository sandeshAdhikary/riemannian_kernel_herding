import torch

def get_submanifold_tensors(x, idx, prod_manifold):
    '''

    Extract the idx-th sub-manifold tensors from a geoopt prod_manifold
    Will also perform the appropriate reshape according to the geoopt product manifold object
    x: The input tensor
    idx: The index
    prod_manifold: The geoopt product manifold object
    '''

    assert idx < len(prod_manifold.manifolds), "The product manifold has less than {} sub-manifolds".format(idx+1)

    return prod_manifold.take_submanifold_value(x, idx)

def geodesic(X, Y, num_manifolds, sub_tensors_fn, dist_fns):
    '''
    Computes distance between tensors X and Y
    sub_tensors_fn: Extract sub-manifold tensors from X and Y
    dist_fns: Distance functions for each manifold
    '''
    assert len(dist_fns) == num_manifolds

    dist = 0
    for idx in range(num_manifolds):
        x, y = sub_tensors_fn(X, idx), sub_tensors_fn(Y, idx)
        dist += dist_fns[idx](x, y)
    return dist

def project(X, num_manifolds, sub_tensors_fn, proj_fns):
    '''
    Project each component of X onto corresponding manifold
    '''

    raise(NotImplementedError)