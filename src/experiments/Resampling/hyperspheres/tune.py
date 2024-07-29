import numpy as np
from src.experiments.Resampling.CoRL.hyperspheres.resample import herd_on_hyperspheres, ot_on_hyperspheres, ot_on_hyperspheres_pf, herd_on_hyperspheres_euclid
import argparse
from src.hyper_param_tuning.random_search import random_search

NUM_PROCESSORS = 8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tune hyperparamters for kernel herding or optimal transport')
    parser.add_argument('resampling_technique')
    args = parser.parse_args()
    resampling_technique = args.resampling_technique
    data_filepath = 'data/weighted_samples_sphere_dist_dims4_seed123_dataseed0.npy'

    if resampling_technique == 'kernel-herding':
        adam_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        adam_epochs = np.array([100]).astype(int)
        bandwidths = [1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['adam_lr', 'adam_epochs', 'bandwidth']
        hyperparam_list = [adam_lrs, adam_epochs, bandwidths]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)
        def run_hyper_fn(x):
            return herd_on_hyperspheres(data_filepath, x)[-1]
        typical_params = {
            "num_data": int(1500),
            "num_resample": int(1500),
        }
        results = random_search(run_hyper_fn, hyperparam_dicts,
                        typical_params=typical_params, num_hypers=num_hyperparams,
                        num_processors=NUM_PROCESSORS)
        np.save('tuning/kernel_herding.npy', results)
    if resampling_technique == 'kernel-herding-euclid':
        adam_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        adam_epochs = np.array([100]).astype(int)
        bandwidths = [1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['adam_lr', 'adam_epochs', 'bandwidth']
        hyperparam_list = [adam_lrs, adam_epochs, bandwidths]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)
        def run_hyper_fn(x):
            return herd_on_hyperspheres_euclid(data_filepath, x)[-1]
        typical_params = {
            "num_data": int(1500),
            "num_resample": int(1500),
        }
        results = random_search(run_hyper_fn, hyperparam_dicts,
                        typical_params=typical_params, num_hypers=num_hyperparams,
                        num_processors=NUM_PROCESSORS)
        np.save('tuning/kernel_herding_euclid.npy', results)

    elif resampling_technique == 'optimal-transport':
        adam_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        adam_epochs = np.array([100]).astype(int)
        ot_regs = [1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['adam_lr', 'adam_epochs', 'ot_reg']
        hyperparam_list = [adam_lrs, adam_epochs, ot_regs]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)

        def run_hyper_fn(x):
            return ot_on_hyperspheres(data_filepath, x)[-1]

        typical_params = {
            "num_data": int(1500),
            "num_resample": int(1500)
        }

        results = random_search(run_hyper_fn, hyperparam_dicts,
                        typical_params=typical_params, num_hypers=num_hyperparams,
                        num_processors=NUM_PROCESSORS)
        np.save('tuning/optimal_transport.npy', results)
        print("Done")
    elif resampling_technique == 'optimal-transport-pf':
        adam_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        adam_epochs = np.array([100]).astype(int)
        ot_regs = [1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['adam_lr', 'adam_epochs', 'ot_reg']
        hyperparam_list = [adam_lrs, adam_epochs, ot_regs]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)

        def run_hyper_fn(x):
            return ot_on_hyperspheres_pf(data_filepath, x)[-1]

        typical_params = {
            "num_data": int(1500),
            "num_resample": int(1500)
        }

        results = random_search(run_hyper_fn, hyperparam_dicts,
                        typical_params=typical_params, num_hypers=num_hyperparams,
                        num_processors=NUM_PROCESSORS)
        np.save('tuning/optimal_transport_pf.npy', results)
        print("Done")
    else:
        raise(ValueError("Unknown resampling technique. Pick from [kernel-herding, optimal-transport]"))

