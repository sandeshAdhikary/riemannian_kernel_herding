import numpy as np
from src.experiments.kernel_ABC.gaussian_estimation.gaussian_covar_estimation.krabc import covar_estimation_herd, covar_estimation_herd_euclid, covar_estimation_herd_cholesky
import argparse
from src.hyper_param_tuning.random_search import random_search_krabc

NUM_PROCESSORS = 4
NUM_HYPERS = 50

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run krabc for gaussian covarianve estimation')
    parser.add_argument('exp_type')
    args = parser.parse_args()
    exp_type = args.exp_type

    # Load dataset
    data = np.load("data/covar_est_data_dims3.npy", allow_pickle=True).item()
    num_processors = NUM_PROCESSORS

    # Load training and test observations. Only use the first of the train/test data for tuning
    y_trains = data['y_train'][0, :]
    y_tests = data['y_test'][0, :]

    if exp_type == 'kernel-herding':
        theta_bandwidth = [None, 1e1, 1.0, 1e-1, 1e-2, 1e-3]
        y_bandwidth = [None, 1e1, 1.0, 1e-1, 1e-2, 1e-3]
        theta_reg = [10.0, 1.0, 1e-1, 1e-3, 1e-9]
        lr = [1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['theta_bandwidth', 'y_bandwidth', 'theta_reg', 'lr']
        hyperparam_list = [theta_bandwidth, y_bandwidth, theta_reg, lr]

        herd_func = covar_estimation_herd

    if exp_type == 'kernel-herding-euclid':
        theta_bandwidth = [None, 1e1, 1.0, 1e-1, 1e-2, 1e-3]
        y_bandwidth = [None, 1e1, 1.0, 1e-1, 1e-2, 1e-3]
        theta_reg = [10.0, 1.0, 1e-1, 1e-3, 1e-9]
        lr = [1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['theta_bandwidth', 'y_bandwidth', 'theta_reg', 'lr']
        hyperparam_list = [theta_bandwidth, y_bandwidth, theta_reg, lr]

        herd_func = covar_estimation_herd_euclid

    if exp_type == 'kernel-herding-cholesky':
        theta_bandwidth = [None, 1e1, 1.0, 1e-1, 1e-2, 1e-3]
        y_bandwidth = [None, 1e1, 1.0, 1e-1, 1e-2, 1e-3]
        theta_reg = [10.0, 1.0, 1e-1, 1e-3, 1e-9]
        lr = [1e-1, 1e-2, 1e-3, 1e-4]
        hyperparam_keys = ['theta_bandwidth', 'y_bandwidth', 'theta_reg', 'lr']
        hyperparam_list = [theta_bandwidth, y_bandwidth, theta_reg, lr]

        herd_func = covar_estimation_herd_cholesky

    num_hyperparams = np.product(np.array([len(x) for x in hyperparam_list]))
    hyperparam_list = np.array(np.meshgrid(*hyperparam_list)).T.reshape(num_hyperparams, -1)
    hyperparam_dicts = []
    for idx in np.arange(len(hyperparam_list)):
        hyperparam_dicts.append(dict(zip(hyperparam_keys, hyperparam_list[idx])))
    hyperparam_dicts = np.array(hyperparam_dicts)

    combined_args = [[y_trains, y_tests, x] for x in hyperparam_dicts]


    def run_hyper_fn(x):
        '''
        x = [y_train y_tests, param]
        '''
        return herd_func(x[0], x[1], x[2])

    typical_params = {
        'dims': data['true_theta'].shape[0],
        'num_iters': 25,
        'theta_bandwidth': 1.0,
        'theta_reg': 1e-3,
        'y_bandwidth': 1.0,
        'num_herd': 100,
        'lr': 1e-3,  # adam LR
        'true_theta': data['true_theta'],
        'true_loc': data['true_loc']
    }
    results = random_search_krabc(run_hyper_fn, combined_args,
                                  typical_params=typical_params, num_hypers=NUM_HYPERS,
                                  num_processors=NUM_PROCESSORS)
    np.save('tuning/{}.npy'.format(exp_type), results)
