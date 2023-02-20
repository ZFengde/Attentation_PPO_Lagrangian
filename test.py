import numpy as np
import torch as th
import torch.nn as nn
import gym
import safety_gym
import turtlebot_env
import time
import stable_baselines3
import stable_baselines3.ppo
import stable_baselines3.ddpg
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn
import TS
import attentation_ppo_lagrangian
import os

# def main():
#     env_id = 'Turtlebot-v1'
#     env = make_vec_env(env_id, n_envs=3, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'obstacle_num': 5})
#     env.reset()
#     a = 1

# if __name__ == '__main__':
#     main()

# env_id = 'Turtlebot-v1'
# obstacle_num = 5

# env = gym.make(env_id, use_gui=True, obstacle_num=obstacle_num)
# logdir = f"Test_log/{env_id}/logs/{int(time.time())}"
# if not os.path.exists(logdir):
#     os.makedirs(logdir)
# model = constrained_rl.FuzzyGNN_PPO_Lagrangian(
# 				'MlpPolicy', 
# 	      		env, 
#                 tensorboard_log=logdir,
# 				verbose=1, 
# 				cost_lim=10., 
# 				net_arch_dim=64, 
# 				obstacle_num=obstacle_num, 
# 				use_constraint=True)
# model.load('FuzzyGNN_PPO_Lagrangian/Turtlebot-v1+n_obstalces=5/models/1676676509/12275712')
# model.test(env)

# env = gym.make('Safexp-PointGoal1-v0')
# obs = env.reset()

# while True:
#     env.render()
#     action = 
#     print(obs)
#     obs, reward, done, info = env.step(action)
#     if done:
#         obs = env.reset()

a = th.tensor([0, .1], dtype=th.float)
b = th.tensor([0, .1], dtype=th.float)

c = th.stack((a, b))
print(c)
print(th.softmax(c, dim=1))



