import numpy as np
import os
from src.manifold_utils.so3_utils import quats_to_rotmat, rotmat_to_quats

method = "SPD_chol"


def std_form_to_sdp(est_standard):
    d, q = est_standard[0:3], est_standard[3:]
    Q = quats_to_rotmat(q.reshape(1, -1)).reshape(3, 3)
    M = Q @ np.diag(d) @ Q.T
    return M

true_params = np.array([0.1, 0.01, 0.001, 0.18257419, 0.36514837, 0.54772256, 0.73029674])
true_params[3:] = true_params[3:]/np.linalg.norm(true_params[3:])
true_mat = std_form_to_sdp(true_params)

folder = 'results/estimations'
all_files = os.listdir(folder)

matrices = []
for file in all_files:
    if file.startswith("est_{}_Jun".format(method)):
        # Get the estimate in standrd form [d1,d2,d3,q1,q2,q3,q4]
        est_standard = np.loadtxt('{}/{}'.format(folder, file))
        M = std_form_to_sdp(est_standard)
        matrices.append(M)

matrices = np.stack(matrices)
mat_mean = np.mean(matrices, axis=0)
mat_std = np.std(matrices, axis=0)

# Write to file
results = np.zeros((9, 4))
results[:, 0] = true_mat.reshape(-1)
results[:, 1] = mat_mean.reshape(-1)
results[:, 2] = mat_std.reshape(-1)
results[:, 3] = (abs(mat_mean - true_mat)/abs(true_mat)).reshape(-1)

np.savetxt('results/estimations/estimation_results_{}.txt'.format(method),
           results, header='true,est,std,percent_error')