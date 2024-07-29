class OrthoHaar():
    '''
    Wrapper around scipy.stats.ortho_group
    O(N) Haar distribution (the only uniform distribution on O(N)).
    '''

    def __init__(self, shape):
        '''
        :param shape: (n,m): Shape of matrices. Require n < m
        '''
