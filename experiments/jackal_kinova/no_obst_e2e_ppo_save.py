import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import os
from datetime import datetime

# specify the ip of the machine running the robot-server
target_machine_ip = "163.180.177.101"  # or other xxx.xxx.xxx.xxx

date = datetime.now()

run_name = "no_obst_e2e_rl_PPO_10"
models_dir = "models/" + run_name
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# initialize environment
env = gym.make("No_Obstacle_Avoidance_Jackal_Kinova_Sim-v0", ip=target_machine_ip)
# env NoObstacleNavigationMir100Sim / ObstacleAvoidanceMir100Sim

env.reset()
# add wrapper for automatic exception handlingz
env = ExceptionHandling(env)

policy_kwarg = dict(activation_fn=nn.ReLU, net_arch=[256, dict(pi=[256, 128], vf=[256, 128])])

# choose and run appropriate algorithm provided by stable-baselines
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwarg,
    verbose=1,
    tensorboard_log="./logs",
)

# model = PPO(
#     "MlpPolicy",
#     env,
#     n_steps=512,
#     verbose=1,
#     tensorboard_log="./logs",
# )


TIMESTEPS = 1000
err_count = 0
i = 1

# model_path = f"{models_dir}/{TIMESTEPS*(i-1)}"
# model = PPO.load(model_path, env=env)
while i < 1000:
    try:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=run_name)
        model.save(f"{models_dir}/{TIMESTEPS*i}")
        print("Error count: ", err_count)
        i = i + 1
    except KeyboardInterrupt as e:
        print(e)
        del env
        exit()

    except:
        err_count = err_count + 1
        print("Got an error while excueting learn(). Retrying...")
        env.close()
        del env
        env = gym.make("Obstacle_Avoidance_Jackal_Kinova_Sim-v0", ip=target_machine_ip)
        env.reset()
        env = ExceptionHandling(env)
        del model

        print("Loading model ", TIMESTEPS * (i - 1))
        model_path = f"{models_dir}/{TIMESTEPS*(i-1)}"
        model = PPO.load(model_path, env=env)

        continue

env.close()
