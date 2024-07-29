import gym
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def transition_fn(env, action, render=False):
    '''
    Takes a step in the environment with the action
    returns observations, and then resets the environment state back to original state
    '''
    orig_state = env.sim.get_state()
    if render: env.render()
    observation, reward, done, info = env.step(action)
    if render: env.render()
    env.sim.set_state(orig_state)
    return observation, reward, done, info

def obj_state_from_env_state(env, full_obj_state=False):
    # This returns just the orientation
    if full_obj_state:
        return env.sim.get_state().qpos[24:31] # obj position (xyz) and orientation (quats)
    else:
        return env.sim.get_state().qpos[27:31] # obj orientation (quats)

def transition_fn_for_obj_orientation(env, q, action):
    '''
    env: the environment
    q: 4d quaternion specifying the object's orientation
    Sets the environment state so that the object's orientation is q
    then takes a step with action
    finally resets the environment back to original state
    '''
    orig_state = env.sim.get_state().flatten()
    # Flattened state is [time (1 item), qpos (38 items), qvel (38 items)]
    new_state = orig_state.copy()
    new_state[28:32] = q # update only the object orientation
    # Update the environment state with the given object orientation
    env.sim.set_state_from_flattened(new_state)
    # Take a step with the given action
    observation, reward, done, info = env.step(action)
    # Reset the state to the original state
    env.sim.set_state_from_flattened(orig_state)
    return observation, reward, done, info


def env_out_proc_fn(env_out, action=None):
    '''
    Process the output from env.step()
    If action is provided, append it to the observations
    '''
    observation, reward, done, info = env_out
    X = observation['achieved_goal'][3:].copy()
    robot_qpos = observation['observation'][0:24].copy()
    robot_qvel = observation['observation'][24:48]
    object_qvel = observation['observation'][48:54].copy()
    touch_values = observation['observation'][54:146].copy()

    Y = np.concatenate([robot_qpos, robot_qvel, object_qvel])

    # if action is None:
    #     Y = np.concatenate([robot_qpos, touch_values])
    # else:
    #     Y = np.concatenate([robot_qpos, touch_values, action])

    return X.reshape(1, -1), Y.reshape(1, -1)

def run_iterations(env, actions, render=False, disp_plot=False):
    env.reset()
    states = np.zeros((actions.shape[0], 4)) # Object's [q1,q2,q3,q4] quaternions

    # num_obs = 4 # Noisy version of the state iteself
    # num_obs = 146 # [robot_qpos, robot_qvel, touch_values, object_qvel]
    # num_obs = 6 # [robot_qvel]
    num_obs = 54 # [robot_qpos, robot_qvel, object_qvel]
    # num_obs = 136  # [robot_qpos, touch_values, actions]
    # [robot_qpos, touch_sensors, actions]
    obs = np.zeros((actions.shape[0], num_obs)) # all robot joint qpos and touch values form sensors

    for idx in np.arange(actions.shape[0]):
        if render: env.render()
        action = actions[idx, :]
        env_out = env.step(action)
        state, observation = env_out_proc_fn(env_out, action=None)
        states[idx, :] = state
        obs[idx, :] = observation
        idx += 1
    env.close()
    if disp_plot:
        fig, axs = plt.subplots(4)
        quart_names = ['q1', 'q2', 'q3', 'q4']
        [axs[idx].plot(states[:, idx], label=quart_names[idx]) for idx in np.arange(4)]
        [axs[idx].set_ylabel('Quaternions') for idx in np.arange(4)]
        [axs[idx].legend() for idx in np.arange(4)]
        [axs[idx].set_ylim(-1.1, 1.1) for idx in np.arange(4)]
        plt.xlabel("Time Steps")
        plt.show()
    return states, obs


if __name__ == "__main__":
    env = gym.make('HandManipulateEggRotateTouchSensors-v0')
    run_iterations(env, render=True)
