import numpy as np
from src.experiments.Resampling.CoRL.spd.resample import get_resamples
import argparse
from src.hyper_param_tuning.random_search import random_search

TUNING_DATA_FILE_PATH = 'data/weighted_samples_spd_dist_dims3_seed123_dataseed0.npy'
NUM_PROCESSORS = 8
NUM_HYPERS = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tune hyperparameters for kernel herding or optimal transport')
    parser.add_argument('resampling_technique')
    args = parser.parse_args()
    resampling_technique = args.resampling_technique
    data_filepath = TUNING_DATA_FILE_PATH

    if resampling_technique == 'kernel-herding':
        adam_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        bandwidths = [1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['adam_lr', 'bandwidth']
        hyperparam_list = [adam_lrs, bandwidths]
        typical_params = {
            "num_data": int(150),
            "num_resample": int(150),
            "adam_epochs": int(100),
            "method": "KernelHerding"
        }
    elif resampling_technique == 'kernel-herding-euclid':
        adam_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        bandwidths = [1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['adam_lr', 'bandwidth']
        hyperparam_list = [adam_lrs, bandwidths]
        typical_params = {
            "num_data": int(150),
            "num_resample": int(150),
            "adam_epochs": int(100),
            "method": "KernelHerdingEuclid"
        }
    elif resampling_technique == 'kernel-herding-chol':
        adam_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        bandwidths = [1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['adam_lr', 'bandwidth']
        hyperparam_list = [adam_lrs, bandwidths]
        typical_params = {
            "num_data": int(150),
            "num_resample": int(150),
            "adam_epochs": int(100),
            "method": "KernelHerdingChol"
        }
    elif resampling_technique == 'optimal-transport':
        adam_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
        ot_regs = [1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['adam_lr', 'ot_reg']
        hyperparam_list = [adam_lrs, ot_regs]
        typical_params = {
            "num_data": int(150),
            "num_resample": int(150),
            "adam_epochs": int(100),
            "method": "OptimalTransport"
        }
    else:
        raise(ValueError("Unknown resampling technique. Pick from [kernel-herding, kernel-herding-euclid, optimal-transport]"))

    num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
    hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
    hyperparam_dicts = []
    for idx in np.arange(len(hyperparam_list)):
        hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
    hyperparam_dicts = np.array(hyperparam_dicts)

    def run_hyper_fn(x):
        return get_resamples(data_filepath, x)[-1]

    results = random_search(run_hyper_fn, hyperparam_dicts,
                            typical_params=typical_params, num_hypers=NUM_HYPERS,
                            num_processors=NUM_PROCESSORS)
    np.save('tuning/kernel_herding_{}.npy'.format(resampling_technique), results)

