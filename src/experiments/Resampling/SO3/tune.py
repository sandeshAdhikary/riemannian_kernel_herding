import numpy as np
from src.experiments.Resampling.CoRL.SO3.resample import herd_resample, ot_resample, ot_resample_pf, ot_resample_pf_cayley, herd_resample_euclid, herd_resample_char
import argparse
from src.hyper_param_tuning.random_search import random_search

NUM_PROCESSORS = 10
TUNING_DATA_FILE_PATH = 'data/weighted_samples_dims3_dataseed0.npy'
NUM_HYPERS = 50

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tune hyperparamters for kernel herding or optimal transport')
    parser.add_argument('resampling_technique')
    args = parser.parse_args()
    resampling_technique = args.resampling_technique
    data_filepath = TUNING_DATA_FILE_PATH

    if resampling_technique == 'kernel-herding':
        bandwidths = [1e3, 200.0, 150.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.5]
        hyperparam_keys = ['bandwidth']
        hyperparam_list = [bandwidths]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)
        def run_hyper_fn(x):
            return herd_resample(data_filepath, x)[-1]
        typical_params = {
            "num_data": int(1500),
            "riemannian_opt": True,
            "num_resample": int(1500),
            "adam_lr": 1e-3,
            "adam_epochs": int(1000),
            "bandwidth": 10.0
        }
        results = random_search(run_hyper_fn, hyperparam_dicts,
                        typical_params=typical_params, num_hypers=num_hyperparams,
                        num_processors=NUM_PROCESSORS)
        np.save('tuning/{}.npy'.format(resampling_technique), results)
    elif resampling_technique == 'kernel-herding-char1':
        bandwidths = [1.0,1.0,1.0,1.0]
        hyperparam_keys = ['bandwidth']
        hyperparam_list = [bandwidths]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)
        def run_hyper_fn(x):
            return herd_resample_char(data_filepath, x, type=1)[-1]
        typical_params = {
            "num_data": int(1500),
            "riemannian_opt": True,
            "num_resample": int(1500),
            "adam_lr": 1e-3,
            "adam_epochs": int(1000),
            "bandwidth": 10.0
        }
        results = random_search(run_hyper_fn, hyperparam_dicts,
                        typical_params=typical_params, num_hypers=num_hyperparams,
                        num_processors=NUM_PROCESSORS)
        np.save('tuning/{}.npy'.format(resampling_technique), results)
    elif resampling_technique == 'kernel-herding-char2':
        bandwidths = [1e3, 200.0, 150.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.5]
        hyperparam_keys = ['bandwidth']
        hyperparam_list = [bandwidths]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)
        def run_hyper_fn(x):
            return herd_resample_char(data_filepath, x, type=2)[-1]
        typical_params = {
            "num_data": int(1500),
            "riemannian_opt": True,
            "num_resample": int(1500),
            "adam_lr": 1e-3,
            "adam_epochs": int(1000),
            "bandwidth": 10.0
        }
        results = random_search(run_hyper_fn, hyperparam_dicts,
                        typical_params=typical_params, num_hypers=num_hyperparams,
                        num_processors=NUM_PROCESSORS)
        np.save('tuning/{}.npy'.format(resampling_technique), results)
    elif resampling_technique == 'kernel-herding-euclid':
        bandwidths = [1e3, 200.0, 150.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.5]
        hyperparam_keys = ['bandwidth']
        hyperparam_list = [bandwidths]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)


        def run_hyper_fn(x):
            return herd_resample_euclid(data_filepath, x)[-1]

        typical_params = {
            "num_data": int(1500),
            "riemannian_opt": False,
            "num_resample": int(1500),
            "adam_lr": 1e-3,
            "adam_epochs": int(1000),
            "bandwidth": 10.0
        }
        results = random_search(run_hyper_fn, hyperparam_dicts,
                                typical_params=typical_params, num_hypers=num_hyperparams,
                                num_processors=NUM_PROCESSORS)
        np.save('tuning/{}.npy'.format(resampling_technique), results)

    elif resampling_technique == 'optimal-transport':
        ot_regs = [1e2, 1e1, 1.0, 1e-1, 5e-1, 5e-2, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['ot_reg']
        hyperparam_list = [ot_regs]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)

        def run_hyper_fn(x):
            return ot_resample(data_filepath, x)[-1]

        typical_params = {
            "num_data": int(1500),
            "riemannian_opt": False,
            "num_resample": int(1500),
            "adam_lr": 1e-3,
            "adam_epochs": int(1000),
            "ot_reg": 1e-3
        }

        results = random_search(run_hyper_fn, hyperparam_dicts,
                        typical_params=typical_params, num_hypers=NUM_HYPERS,
                        num_processors=NUM_PROCESSORS)
        np.save('tuning/{}.npy'.format(resampling_technique), results)

    elif resampling_technique == 'optimal-transport-pf':
        ot_regs = [1e2, 1e1, 1.0, 1e-1, 5e-1, 5e-2, 1e-2, 1e-3, 1e-4]
        adam_lrs = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['ot_reg', 'adam_lr']
        hyperparam_list = [ot_regs, adam_lrs]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)


        def run_hyper_fn(x):
            return ot_resample_pf(data_filepath, x)[-1]


        typical_params = {
            "num_data": int(1500),
            "riemannian_opt": False,
            "num_resample": int(1500),
            "adam_lr": 1e-3,
            "adam_epochs": int(1000),
            "ot_reg": 1e-3
        }

        results = random_search(run_hyper_fn, hyperparam_dicts,
                                typical_params=typical_params, num_hypers=NUM_HYPERS,
                                num_processors=NUM_PROCESSORS)
        np.save('tuning/{}.npy'.format(resampling_technique), results)

    elif resampling_technique == 'optimal-transport-pf-cayley':
        ot_regs = [1e2, 1e1, 1.0, 1e-1, 5e-1, 5e-2, 1e-2, 1e-3, 1e-4]
        adam_lrs = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['ot_reg', 'adam_lr']
        hyperparam_list = [ot_regs, adam_lrs]
        num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
        hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
        hyperparam_dicts = []
        for idx in np.arange(len(hyperparam_list)):
            hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
        hyperparam_dicts = np.array(hyperparam_dicts)


        def run_hyper_fn(x):
            return ot_resample_pf_cayley(data_filepath, x)[-1]


        typical_params = {
            "num_data": int(1500),
            "riemannian_opt": True,
            "num_resample": int(1500),
            "adam_lr": 1e-3,
            "adam_epochs": int(1000),
            "ot_reg": 1e-3
        }
        results = random_search(run_hyper_fn, hyperparam_dicts,
                                typical_params=typical_params, num_hypers=NUM_HYPERS,
                                num_processors=NUM_PROCESSORS)
        np.save('tuning/{}.npy'.format(resampling_technique), results)

    else:
        raise(ValueError("Unknown resampling technique. Pick from [kernel-herding, kernel-herding-euclid, optimal-transport, optimal-transport-pf, optimal-transport-pf-cayley]"))

