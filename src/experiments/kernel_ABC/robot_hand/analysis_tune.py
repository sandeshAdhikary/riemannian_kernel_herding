import numpy as np
import matplotlib.pyplot as plt
import torch
from src.experiments.kernel_ABC.robot_hand.krabc import inertia_estimate_experiment
import pandas as pd
import altair as alt
import argparse

seeds = np.arange(3, 15)
methods = ['SPD_euclid']
# methods = ["SPD", "SPD_euclid", "SPD_chol"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run krabc for gaussian covarianve estimation')
    parser.add_argument('numherd')
    args = parser.parse_args()
    numherd = args.numherd

    tune_folder = 'tuning/saved/numherd{}'.format(numherd)
    results_folder = 'results/saved/numherd{}'.format(numherd)
    for method in methods:
        print(results_folder)
        print("Running method: {} for numherd {}".format(method, numherd))

        results = np.load('{}/{}.npy'.format(tune_folder,method), allow_pickle=True)
        params = [res['hyperparamters'] for res in results]
        val_losses = [res['val_loss'] for res in results]
        best_idx = np.argmin(val_losses)
        best_params = params[best_idx]
        best_loss = val_losses[best_idx]
        print("Best Loss: {}".format(best_loss))
        print("Best Params: {}".format(best_params))


        plt.plot(val_losses)
        plt.ylabel("Sampilng Errors")
        plt.xlabel("Hyperparameter configuration number")
        plt.savefig('{}/{}.png'.format(tune_folder, method))
        plt.close()

        tuning_out = {
            'best_val_loss': best_loss,
            'best_val_params': best_params
        }

        np.save('{}/{}_best_params.npy'.format(tune_folder, method), tuning_out)

        # Plot altair chart
        data = pd.DataFrame(params)
        data['val_loss'] = val_losses
        data[data.columns] = data[data.columns].astype(float)

        if method in ["SPD", "SPD_chol", "SPD_euclid"]:
            repeat_cols = ['theta_bandwidth', 'theta_reg', 'lr']
        elif method in ["quat","quat_euclid"]:
            repeat_cols = ['theta_bandwidth_diags', 'theta_bandwidth_quats', 'theta_reg', 'lr']
        chart = alt.Chart(data).mark_circle(opacity=.5, color="steelblue").encode(
            x=alt.X(alt.repeat('repeat'), type='ordinal'),
            y=alt.Y("val_loss:Q")
        )
        chart_mean = alt.Chart(data).mark_circle(color="red").encode(
            x=alt.X(alt.repeat('repeat'), type='ordinal'),
            y=alt.Y("mean(val_loss):Q")
        )

        fullchart = (chart + chart_mean).repeat(
            repeat=repeat_cols,
            columns=3
        )

        fullchart.save("{}/chart_{}.html".format(tune_folder, method))

        ### Test the best params on test datasets and get final performance
        assert method in ["quat", "quat_euclid", "SPD", "SPD_euclid", "SPD_chol"]

        # Load the tuning datasets
        y_trains, y_tests, true_thetas, actions = [], [], [], []
        for seed in seeds:
            data = np.load('data/test_data/data_seed{}.npy'.format(seed), allow_pickle=True).item()
            true_thetas.append(torch.from_numpy(data['true_theta']))
            y_trains.append(torch.from_numpy(data['observations']).reshape(1, -1))
            y_tests.append(torch.from_numpy(data['observations']).reshape(1, -1))
            actions.append(data['actions'])

        # Get mean error across all rounds
        best_params['num_iters'] = 20 # Run for longer iterations
        sim_errs_full, est_errs_full = inertia_estimate_experiment(y_trains, y_tests, true_thetas,
                                                                   actions, best_params, method, plot=True,
                                                                   full_output=True)
        sim_errs_full, est_errs_full = np.array(sim_errs_full), np.array(est_errs_full)
        sim_errs_mean = np.mean(sim_errs_full, axis=0)
        sim_errs_std = np.std(sim_errs_full, axis=0)

        est_errs_mean = np.mean(est_errs_full, axis=0)
        est_errs_std = np.std(est_errs_full, axis=0)

        # Plot errors
        fig, axs = plt.subplots(2)
        axs[0].plot(np.arange(len(est_errs_mean)), est_errs_mean)
        axs[0].fill_between(np.arange(len(est_errs_mean)),
                            est_errs_mean-est_errs_std,
                            est_errs_mean+est_errs_std, alpha=0.3)
        axs[0].set_ylabel("Estimation Errors")
        axs[1].plot(np.arange(len(sim_errs_mean)), sim_errs_mean)
        axs[1].fill_between(np.arange(len(sim_errs_mean)),
                            sim_errs_mean-sim_errs_std,
                            sim_errs_mean+sim_errs_std, alpha=0.3)
        axs[1].set_ylabel("Simulation Errors")
        plt.savefig('{}/test_results_{}.png'.format(results_folder, method))

        full_output = [sim_errs_full, est_errs_full]
        np.save('{}/test_results_{}.npy'.format(results_folder, method), full_output)
