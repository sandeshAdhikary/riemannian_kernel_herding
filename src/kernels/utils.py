import numpy as np
from scipy.spatial.distance import pdist

def median_trick_bandwidth(data):
    '''
    Get bandwidth for an RBF kernel via the median trick/heuristic
    '''

    # Median trick used in svgd
    # sq_dist = pdist(data)
    # pairwise_dists = squareform(sq_dist) ** 2
    #
    # h = np.median(pairwise_dists)
    # h = np.sqrt(0.5 * h / np.log(data.shape[0] + 1))

    # simple median
    dist = pdist(data)
    h = np.sqrt(np.median(dist) / 2)

    return h