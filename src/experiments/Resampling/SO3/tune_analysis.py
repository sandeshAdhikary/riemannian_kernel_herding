import numpy as np

method = 'kernel-herding-char1'

tune_res = np.load('tuning/{}.npy'.format(method), allow_pickle=True).item()

best_idx = np.argmin(tune_res['val_loss'])
best_params = tune_res['hyperparameters'][best_idx]
best_loss = tune_res['val_loss'][best_idx]

print("Best val_loss: {}".format(best_loss))
print("Best params: {}".format(best_params))
