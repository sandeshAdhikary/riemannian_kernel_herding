import numpy as np
import torch
from src.experiments.kernel_ABC.robot_hand.krabc import inertia_estimate_experiment
import argparse

NUM_HYPERS = 1000
NUM_ITERS = 10
NUM_EPOCHS = 100
NUM_PROCESSORS = 8
NUM_HERD = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run krabc for gaussian covarianve estimation')
    parser.add_argument('method')
    args = parser.parse_args()
    method = args.method

    assert method in ["quat", "quat_euclid", "SPD", "SPD_euclid", "SPD_chol"]

    # Load the tuning datasets
    seeds = [0, 1, 2]
    y_trains, y_tests, true_thetas, actions = [], [], [], []
    for seed in seeds:
        data = np.load('data/tune_data/data_seed{}.npy'.format(seed), allow_pickle=True).item()
        true_thetas.append(torch.from_numpy(data['true_theta']))
        y_trains.append(torch.from_numpy(data['observations']).reshape(1, -1))
        y_tests.append(torch.from_numpy(data['observations']).reshape(1, -1))
        actions.append(data['actions'])

    if method in ["quat", "quat_euclid"]:
        theta_bandwidths_diags = [10.0, 1.0, 1e-1, 1e-2, 1e-3]
        theta_bandwidths_quats = [10.0, 1.0, 1e-1, 1e-2, 1e-3]
        theta_reg = [1e-3, 1e-6, 1e-9]
        lr = [1.0, 1e-1, 1e-2, 1e-3]
        hyperparam_keys = ['theta_bandwidth_diags', 'theta_bandwidth_quats', 'theta_reg', 'lr']
        hyperparam_list = [theta_bandwidths_diags, theta_bandwidths_quats, theta_reg, lr]

        typical_params = {
            'num_herd': NUM_HERD,
            'num_epochs': NUM_EPOCHS,
            'num_iters': NUM_ITERS,
            'num_processors': NUM_PROCESSORS
        }

    elif method in ["SPD", "SPD_euclid", "SPD_chol"]:
        theta_bandwidth = [10, 1.0, 1e-1, 1e-2, 1e-3]
        theta_reg = [1e-3, 1e-6, 1e-9]
        lr = [1.0, 1e-1, 1e-2, 1e-3]
        hyperparam_keys = ['theta_bandwidth', 'theta_reg', 'lr']
        hyperparam_list = [theta_bandwidth, theta_reg, lr]

        typical_params = {
            'num_herd': NUM_HERD,
            'num_epochs': NUM_EPOCHS,
            'num_iters': NUM_ITERS,
            'num_processors': NUM_PROCESSORS
        }
    else:
        raise(ValueError("Unknown sampling method"))

    num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
    hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
    hyperparam_dicts = []
    for idx in np.arange(len(hyperparam_list)):
        hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
    hyperparam_dicts = np.array(hyperparam_dicts)

    # Pick a random subset of hyperparameters
    actual_hypers = min(NUM_HYPERS, len(hyperparam_dicts))
    print("Tuning over {} hyperparameter settings".format(actual_hypers))
    idxes = np.random.choice(np.arange(len(hyperparam_dicts)), replace=False, size=actual_hypers)
    selected_hypers = hyperparam_dicts[idxes]

    # Create a list of typical_params, and update with new new hyperparam values
    full_hypers = [typical_params.copy() for _ in np.arange(actual_hypers)]
    [x.update(y) for (x, y) in zip(full_hypers, selected_hypers)]

    tune_results = []
    for hyperparam in full_hypers:
        try:
            err = inertia_estimate_experiment(y_trains, y_tests, true_thetas,
                                        actions, hyperparam, method, plot=False)
            tune_results.append({'val_loss': err, 'hyperparamters': hyperparam})
            np.save('tuning/{}.npy'.format(method), tune_results)
        except Exception as e:
            print(e)




