# import mj_envs
# from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0 as ENV
# from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0 as ENV
# from mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0 as ENV
from mj_envs.hand_manipulation_suite.hand_v0 import HandEnvV0 as ENV
# from mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0 as ENV
import numpy as np

env = ENV()
# body_idx = env.sim.model.body_name2id('forearm')
body_idx = env.sim.model.body_name2id('palm')

# env.sim.model.body_inertia[body_idx] = [1.0,0.0,0.0]
# env.sim.model.body_inertia[body_idx] = [0.0,1.0,0.0]
# env.sim.model.body_inertia[body_idx] = [0.0,0.0,1.0]
# env.sim.model.body_inertia[body_idx] = [1.0,1.0,1.0]
# env.sim.model.body_inertia[body_idx] = [0.1,0.1,0.1]
# env.sim.model.body_inertia[body_idx] = [10, 5, 1]
# env.sim.model.body_inertia[body_idx] = env.sim.model.body_inertia[body_idx]*10
#
# for idx in range(env.sim.model.body_iquat.shape[0]):
#     env.sim.model.body_iquat[body_idx,:] = np.random.random(4)
#     env.sim.model.body_iquat[body_idx] = env.sim.model.body_iquat[idx]/np.linalg.norm(env.sim.model.body_iquat[body_idx])

# env.sim.model.body_iquat[body_idx] = [1.0,0,0,0]
# env.sim.model.body_iquat[body_idx] = [0.0,1.0,0,0]
# env.sim.model.body_iquat[body_idx] = [0,1,0,0]
# env.sim.model.body_iquat[body_idx] = [1.0,0,0,0]
# env.sim.model.body_iquat[body_idx] = [20.0,10,5,1]
# env.sim.model.body_iquat[body_idx] = [1,5,10,20]
# env.sim.model.body_iquat[body_idx] = np.random.random(env.sim.model.body_iquat[body_idx].shape)
# env.sim.model.body_iquat[body_idx] = np.array(env.sim.model.body_iquat[body_idx])
# env.sim.model.body_iquat[body_idx] = env.sim.model.body_iquat[body_idx]/np.linalg.norm(env.sim.model.body_iquat[body_idx])
# for idx in range(env.sim.model.body_inertia.shape[0]):
#     env.sim.model.body_inertia[idx,:] = np.random.random(3)*1



episodes = 10
num_iters = 200

np.random.seed(453)
env.sim.model.body_iquat[body_idx] = np.random.random(env.sim.model.body_iquat[body_idx].shape)
env.sim.model.body_iquat[body_idx] = np.array(env.sim.model.body_iquat[body_idx])
env.sim.model.body_iquat[body_idx] = env.sim.model.body_iquat[body_idx] / np.linalg.norm(env.sim.model.body_iquat[body_idx])


# actions = np.random.random((num_iters, env.action_space.sample().shape[0]))
actions = np.zeros((num_iters,env.action_space.sample().shape[0]))
env.seed(10)
for ep in range(episodes):

    # env.reset()
    for idx in range(num_iters):
        env.mj_render()
        # action = env.action_space.sample()
        action = actions[idx, :]
        if idx < num_iters/4:
            action[1] = 1
        elif idx > num_iters/4 and idx < num_iters/2:
            action[1] = 0
        elif idx > num_iters/2 and idx < 3*num_iters/4:
            action[1] = 1
        else:
            action[1] = 0
        ob, reward, done, goal_achieved = env.step(action)

print("Done")