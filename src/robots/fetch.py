import gym
from gym.envs.robotics import fetch_env
from gym.envs.robotics.fetch import push
from gym.envs.robotics.hand import manipulate
from gym.envs.robotics.hand import manipulate_touch_sensors
import numpy as np
SEED = 0


# env = gym.make('HandManipulateEgg-v0')
# env = gym.make('HandManipulatePen-v0')
import os
from gym import utils
from gym.envs.robotics import fetch_env


# # Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

env = FetchPushEnv()
np.random.seed(SEED)
env.seed(SEED)

env.reset()
outer_iters = 10
inner_iters = 100
pos_delta = 0.2
num_obs = 25
obs = np.zeros((outer_iters*inner_iters, num_obs))
idx = 0
# "elbow_flex_link" == env.sim.model.body_names[15]
# "wrist_flex_link" == 17
# "gripper_link" == 19
body_idx = 15



# for idx in range(env.sim.model.body_inertia.shape[0]):
env.sim.model.body_inertia[body_idx, :] = np.random.random(3)*1e5
# env.sim.model.body_iquat[body_idx,:] = np.array([1.0,0.0,0.0,1.0])
# env.sim.model.body_iquat[body_idx,:] = env.sim.model.body_iquat[15,:]/np.linalg.norm(env.sim.model.body_iquat[15,:])

for main_iter in range(outer_iters):
    for i_ep in range(inner_iters):
        env.render()
        if i_ep < int(inner_iters/2):
            observation, reward, done, info = env.step(np.array([0,0,pos_delta,0.]))
        else:
            observation, reward, done, info = env.step(np.array([0,0,-pos_delta,0.]))
        # observation, reward, done, info = env.step(env.action_space.sample())
        # obs[idx,:] = observation['observation']
        idx += 1
env.close()

# Collect all gripper observations
## Note: We can also obtain all joint position and velocities, not just those for gripper
# grip_pos = obs[:, 0:3]
# grip_state = obs[:, 9:11]
# grip_velp = obs[:, -5:-2]
# grip_vel = obs[:, -2:]
#
# print("Done")