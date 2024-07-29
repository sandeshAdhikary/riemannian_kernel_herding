from mj_envs.hand_manipulation_suite.hand_v0 import HandEnvV0
import numpy as np
import matplotlib.pyplot as plt

# NOTES:
# All 30 joints are listed here: self.sim.model.joint_names
# The first 24 elements in env.step seems to be pulling the positions
# of the 24 joint of the hand, the last 6 ommitted ones are object
# joints ['OBJTx', 'OBJTy', 'OBJTz', 'OBJRx', 'OBJRy', 'OBJRz']
# The joints 'WRJ0' and 'WRJ1' should be the wrist joints
# These show the most discrenable oscillations based on our controller

ENV = HandEnvV0()
ENV.seed(1234)
ENV.reset()
PALM_IDX = ENV.sim.model.body_name2id('palm') # For inertia params
JOINT_NAMES = ['WRJ0', 'WRJ1']
JOINT_IDS = [ENV.sim.model.joint_name2id(x) for x in JOINT_NAMES] # For observations
NUM_SEEDS = 25
NOISE_SCALE = 1

def run_hand_simulation(input_env, theta, actions, obs_joint_idxes=JOINT_IDS, render=False):
    '''
    Theta: 7D vector. 1st 3 entries are positive diag elements, last 4 entries are quaternions
    '''

    input_env.reset()

    # Save the original inertia values
    orig_inertia = input_env.sim.model.body_inertia[PALM_IDX].copy()
    orig_iquat = input_env.sim.model.body_iquat[PALM_IDX].copy()

    # Update inertial values
    input_env.sim.model.body_inertia[PALM_IDX] = theta[0:3]
    input_env.sim.model.body_iquat[PALM_IDX] = theta[3:]

    # Run simulation
    observations = np.zeros((actions.shape[0], len(obs_joint_idxes)))
    for action_id in range(actions.shape[0]):
        if render:
            input_env.mj_render()
        action = actions[action_id, :]
        input_env.step(action)
        for obs_id in np.arange(len(obs_joint_idxes)):
            observations[action_id, obs_id] = input_env.sim.data.qpos[obs_id]
        # observations.append([input_env.sim.data.qpos[obs_joint_idx]])
    observations = np.reshape(observations, np.product(observations.shape), "F")

    # Reset inertia values to original
    input_env.sim.model.body_inertia[PALM_IDX] = orig_inertia
    input_env.sim.model.body_iquat[PALM_IDX] = orig_iquat

    input_env.reset()
    return observations


def gen_actions(noise_level=1.0, num=1):
    '''
    noise_level: multiple of uniformly distributed noise added to controls
    num: How many sequences of actions to generate
    '''

    # First create a sequence of baseline controls:
    # We're moving the hand diagonally: left to right, then right to left
    top_actions = np.zeros((int(num_iters/(4*num_cycles)), env.action_space.sample().shape[0]))
    bottom_actions = np.zeros((int(num_iters/(4*num_cycles)), env.action_space.sample().shape[0]))
    left_actions = np.zeros((int(num_iters/(4*num_cycles)), env.action_space.sample().shape[0]))
    right_actions = np.zeros((int(num_iters/(4*num_cycles)), env.action_space.sample().shape[0]))
    top_actions[:, 0] = 1
    top_actions[:, 1] = 1
    bottom_actions[:, 0] = -1
    bottom_actions[:, 1] = -1
    left_actions[:, 0] = 1
    left_actions[:, 1] = -1
    right_actions[:, 0] = -1
    right_actions[:, 1] = 1

    actions = np.vstack([top_actions, bottom_actions, left_actions, right_actions])
    actions = np.tile(actions, (num_cycles, 1))

    # Now set up multiple versions of noisy actions
    noisy_actions = np.zeros((num, actions.shape[0], actions.shape[1]))
    for act_num in np.arange(num):
        noisy = actions.copy()
        noisy[noisy > 0] -= np.abs(np.random.random(noisy[noisy > 0].shape)*noise_level)
        noisy[noisy < 0] += np.abs(np.random.random(noisy[noisy < 0].shape) * noise_level)
        noisy_actions[act_num, :] = noisy

    return noisy_actions

if __name__ == "__main__":

    # Set up the environment
    seed = 1234
    np.random.seed(seed)
    env = HandEnvV0()
    env.seed(seed)
    env.reset()

    joint_names = JOINT_NAMES
    joint_ids = [env.sim.model.joint_name2id(x) for x in joint_names]

    # Set the inertia thetas
    theta_inertia = [0.1, 0.01, 0.001]
    theta_iquats = np.array([1, 2, 3, 4])
    theta_iquats = theta_iquats/np.linalg.norm(theta_iquats)
    theta = np.hstack([theta_inertia, theta_iquats])

    # Set up the number of iterations
    num_cycles = 4
    assert num_cycles % 4 == 0, "Pick num_cycles divisible by 4"
    num_iters = num_cycles * 10

    # Set up the actions
    ## We're moving the wrist in diagonal directions alternatively
    noisy_actions = gen_actions(noise_level=NOISE_SCALE, num=NUM_SEEDS)

    # Run simulation
    num_obs = len(joint_ids) * noisy_actions.shape[1]
    obs_noisy = np.zeros((NUM_SEEDS, num_obs))
    for seed in np.arange(NUM_SEEDS):
        observations = run_hand_simulation(env, theta,
                                           noisy_actions[seed,:],
                                           obs_joint_idxes=joint_ids,
                                           render=False)
        obs_noisy[seed, :] = observations

        data = {
            'seed': seed,
            'observations': observations,
            'actions': noisy_actions[seed, :],
            'true_theta': theta,
            'palm_body_id': PALM_IDX,
            'obs_joint_names': joint_names,
            'obs_joint_ids': joint_ids
        }

        np.save('data/data_seed{}.npy'.format(seed), data)

    for seed in np.arange(NUM_SEEDS):
        plt.plot(obs_noisy[seed,:], label='Seed:{}'.format(seed))
    plt.legend()
    plt.savefig('data/obs_traj.png')
    plt.close()


    mean_obs = np.mean(obs_noisy, axis=0)
    std_obs = np.std(obs_noisy, axis=0)
    plt.plot(np.arange(len(mean_obs)), mean_obs)
    plt.fill_between(np.arange(len(mean_obs)),
                     mean_obs+std_obs, mean_obs-std_obs,alpha=0.3)
    plt.savefig('data/obs_traj.png')
    plt.close()
    #
