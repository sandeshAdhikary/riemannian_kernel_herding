import numpy as np
from multiprocessing import Pool, cpu_count

def random_search(run_hyper_fn, hyperparam_dicts, typical_params, num_hypers, num_processors=1):
    '''
    run_hyper_fn(x): Runs the hyperparameter setting x and returns validation loss
    hyperparam_dicts: A list of dicts with hyperparameters to tune
    typical_params: A dictionary specifying typical hyperparameters for the experiment
                    Any key that is present in elements in hyperparam_dicsts will be replaced, the others retained
    num_hypers: The number of hyperparameters to actually run
    num_processors: Number of processorts to use. If -1, use as many as there are available
    '''

    # Pick a random subset of hyperparameters
    actual_hypers = min(num_hypers, len(hyperparam_dicts))
    print("Tuning over {} hyperparameter settings".format(actual_hypers))
    idxes = np.random.choice(np.arange(len(hyperparam_dicts)), replace=False, size=actual_hypers)
    selected_hypers = hyperparam_dicts[idxes]

    # Create a list of typical_params, and update with new new hyperparam values
    full_hypers = [typical_params.copy() for _ in np.arange(actual_hypers)]
    [x.update(y) for (x, y) in zip(full_hypers, selected_hypers)]


    with Pool(min(num_processors, cpu_count() - 1)) as pool:# Use one less than total cpus to prevent freeze
        val_loss = pool.map(func=run_hyper_fn, iterable=full_hypers)

    results = {
        'hyperparameters': full_hypers,
        'val_loss': val_loss
    }
    return results

def random_search_krabc(run_hyper_fn, all_args, typical_params, num_hypers, num_processors=1):
    '''
    run_hyper_fn(x): Runs the hyperparameter setting x and returns validation loss
    hyperparam_dicts: A list of dicts with hyperparameters to tune
    typical_params: A dictionary specifying typical hyperparameters for the experiment
                    Any key that is present in elements in hyperparam_dicsts will be replaced, the others retained
    num_hypers: The number of hyperparameters to actually run
    num_processors: Number of processorts to use. If -1, use as many as there are available
    '''

    y_trains = [x[0] for x in all_args]
    y_tests = [x[1] for x in all_args]
    hyperparam_dicts = np.array([x[2] for x in all_args], dtype=object)

    # Pick a random subset of hyperparameters
    actual_hypers = min(num_hypers, len(hyperparam_dicts))
    print("Tuning over {} hyperparameter settings".format(actual_hypers))
    idxes = np.random.choice(np.arange(len(hyperparam_dicts)), replace=False, size=actual_hypers)
    selected_hypers = hyperparam_dicts[idxes]

    # Create a list of typical_params, and update with new new hyperparam values
    full_hypers = [typical_params.copy() for _ in np.arange(actual_hypers)]
    [x.update(y) for (x, y) in zip(full_hypers, selected_hypers)]

    full_args = [[x, y, z] for (x, y, z) in zip(y_trains[0:actual_hypers],
                                            y_tests[0:actual_hypers],
                                            full_hypers)]

    with Pool(min(num_processors, cpu_count() - 1)) as pool:# Use one less than total cpus to prevent freeze
        val_loss = pool.map(func=run_hyper_fn, iterable=full_args)

    results = {
        'hyperparameters': full_hypers,
        'val_loss': val_loss
    }
    return results
