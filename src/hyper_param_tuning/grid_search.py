import numpy as np
from time import time

def grid_search(hyperparameters, run_then_return_val_loss, num_iters, verbose=True):
    '''
    :param hyperparameters: List of dictionaries of hyperparameter configurations
    :param run_then_return_val_loss(num_iters=r,hyperparameters=t): function to train using hyperparam and return val loss
    :param num_iters: Number of iterations to run run_then_return_val_loss
    :return:
    '''

    val_losses = []
    best_val_loss = np.inf
    for idx, hyp in enumerate(hyperparameters):
        start = time()
        try:
            print('Trying param config {} of {}'.format(idx, len(hyperparameters)))
            val_loss = run_then_return_val_loss(num_iters = num_iters, hyperparameters=hyp)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_param = hyp
                best_val_loss = val_loss
                if verbose:
                    print('Best val loss: {}'.format(val_loss))
                    print('Best params: {}'.format(best_param))
        except:
            print('Could not evaluate hyperparam. Moving on to next one...')
            continue

        if verbose: print('\tCompleted param config {} of {} in {:.3} secs'.format(idx, len(hyperparameters),
                                                                                 time()-start))

    sorted_idx = np.argsort(val_losses)
    out_hypers = np.array(hyperparameters)[sorted_idx]
    out_losses = np.array(val_losses)[sorted_idx]

    print('Best val loss: {}'.format(out_losses[0]))
    print('Best params: {}'.format(out_hypers[0]))

    return out_hypers, out_losses



