import gym
import safety_gym
import turtlebot_env
import attentation_ppo_lagrangian

import os
import time
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def main(
		env_id, 
		algo, 
		policy_type, 
		n_envs, 
		iter_num, 
		seed, 
		cost_lim, 
		net_arch_dim, 
		obstacle_num, 
		use_constraint, 
		test):

	algo_name = algo

	algo = eval('attentation_ppo_lagrangian.'+algo)
	if 'Turtlebot' in env_id:
		# env_kwargs = {'obstacle_num': obstacle_num, 'use_gui': True}
		env_kwargs = {'obstacle_num': obstacle_num}
	else:
		env_kwargs = None

	env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
	# make experiment directory
	logdir = f"{algo_name}/{env_id}+n_obstalces={obstacle_num}/logs/{int(time.time())}/"
	modeldir = f"{algo_name}/{env_id}+n_obstalces={obstacle_num}/models/{int(time.time())}/"

	if not os.path.exists(modeldir):
		os.makedirs(modeldir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)
		
	model = algo(
				policy_type, 
	      		env, 
				verbose=1, 
				tensorboard_log=logdir, 
				cost_lim=cost_lim, 
				net_arch_dim=net_arch_dim, 
				obstacle_num=obstacle_num, 
				use_constraint=use_constraint)

	for i in range(iter_num):
		model.learn(reset_num_timesteps=False, tb_log_name=f"{algo_name}")
		model.save(modeldir, f'{i * n_envs * model.n_steps}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='Turtlebot-v1') # 'Turtlebot-v1''Safexp-PointGoal1-v0'
    parser.add_argument('--algo', type=str, default='Atte_PPO_Lagrangian') # 'PPO_Lagrangian'
    parser.add_argument('--policy_type', type=str, default='MlpPolicy')
    parser.add_argument('--n_envs', type=int, default=3)
    parser.add_argument('--iter_num', type=int, default=1000) # Total_timestep = iter_num * n_envs * n_steps, here is 200 * 3 * 20480 = 1.2e7
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--net_arch_dim', type=int, default=64)
    parser.add_argument('--obstacle_num', type=int, default=4)
    parser.add_argument('--cost_lim', type=float, default=0.) # 0.1 per time step
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--use_constraint', action='store_true') # if no action, or said default if False, otherwise it's True
    args = parser.parse_args()
    args.use_constraint = True
    
    main(
	    args.env_id, 
		args.algo, 
		args.policy_type, 
		args.n_envs, 
		args.iter_num, 
		args.seed,
		args.cost_lim,
		args.net_arch_dim,
		args.obstacle_num,
		args.use_constraint,
		args.test)
