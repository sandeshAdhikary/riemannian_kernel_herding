import numpy as np
import matplotlib.pyplot as plt
import torch
from src.experiments.Resampling.CoRL.spd.resample import get_resamples
from multiprocessing import Pool, cpu_count
import pandas as pd
import altair as alt

NUM_PROCESSORS = 8
# methods = ["kernel_herding_kernel-herding", "kernel_herding_kernel-herding-euclid",
#            "kernel_herding_kernel-herding-chol", "kernel_herding_optimal-transport"]
methods = ["kernel_herding_kernel-herding-chol"]
print(methods)
seeds = [1, 2, 3, 4, 5]

for method in methods:
    print("Running method: {}".format(method))

    results = np.load('tuning/{}.npy'.format(method), allow_pickle=True).item()
    params = results['hyperparameters']
    val_losses = results['val_loss']
    best_idx = np.argmin(val_losses)
    best_params = params[best_idx]
    best_loss = val_losses[best_idx]
    print("Best Loss: {}".format(best_loss))
    print("Best Params: {}".format(best_params))


    plt.plot(val_losses)
    plt.ylabel("Sampilng Errors")
    plt.xlabel("Hyperparameter configuration number")
    plt.savefig('tuning/{}.png'.format(method))
    plt.close()

    tuning_out = {
        'best_val_loss': best_loss,
        'best_val_params': best_params
    }

    np.save('tuning/{}_best_params.npy'.format(method), tuning_out)

    # Plot altair chart
    data = pd.DataFrame(params)
    data['val_loss'] = val_losses
    data = data.drop('method',axis=1)
    data[data.columns] = data[data.columns].astype(float)

    repeat_cols = ['adam_lr','bandwidth']
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


    fullchart.save("tuning/chart_{}.html".format(method))

    # ### Test the best params on test datasets and get final performance
    data_filepaths = ['data/weighted_samples_spd_dist_dims3_seed123_dataseed{}.npy'.format(seed) for seed in seeds]
    exp_inputs = [[data_filepath, best_params] for data_filepath in data_filepaths]


    num_processors = NUM_PROCESSORS
    with Pool(min(num_processors, cpu_count() - 1)) as pool:  # Use one less than total cpus to prevent freeze
        results = pool.starmap(func=get_resamples, iterable=exp_inputs)

    np.save('results/resampling_exp_errs_{}.npy'.format(best_params['method']), results)

