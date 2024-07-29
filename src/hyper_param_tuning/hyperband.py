import numpy as np

def hyperband(get_random_hyperparameter_configuration,
              run_then_return_val_loss,
              max_iter=81,
              eta=3,
              min_iter=1,
              verbose=True):
    '''
    :param get_random_hyperparameter_configuration: function that samples a hyperparam config
    :param run_then_return_val_loss(num_iters=r,hyperparameters=t): function to train using hyperparam and return val loss
    :param max_iter: maximum iterations/epochs per configuration
    :param eta: defines downsampling rate (default=3)
    :return:
    '''

    logeta = lambda x: np.log(x)/np.log(eta)
    s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
    B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
    for s in reversed(range(s_max+1)):
        if verbose:
            print('Outer Loop: {} of {}'.format(s_max+1-s, s_max+1))

        n = int(np.ceil(int(B/max_iter/(s+1))*eta**s)) # initial number of configurations
        r = max_iter*eta**(-s) # initial number of iterations to run configurations for

        #### Begin Finite Horizon Successive Halving with (n,r)
        T = [ get_random_hyperparameter_configuration() for i in range(n) ]
        for i in range(s+1):
            if verbose:
                print('\t\tInner Loop: {} of {}'.format(i, s + 1))
            # Run each of the n_i configs for r_i iterations and keep best n_i/eta
            n_i = n*eta**(-i)
            r_i = max(r*eta**(i), min_iter)
            val_losses = [ run_then_return_val_loss(num_iters=np.int(r_i), hyperparameters=t) for t in T ]

            if verbose:
                print('Lowest Val Loss: {}'.format(np.min(val_losses)))
                print('Best Params:{}'.format(T[np.argmin(val_losses)]))

            T = [T[i] for i in np.argsort(val_losses)[0:int(n_i / eta)]]

        #### End Finite Horizon Successive Halving with (n,r)

    val_losses = np.sort(val_losses)
    return T, val_losses