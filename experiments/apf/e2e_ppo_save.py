import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import os
from datetime import datetime

# specify the ip of the machine running the robot-server
target_machine_ip = "163.180.177.101"  # or other xxx.xxx.xxx.xxx

date = datetime.now()

models_dir = "models/e2e_rl_PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# initialize environment
env = gym.make("Obstacle_Avoidance_Jackal_Kinova_Sim-v0", ip=target_machine_ip)
# env NoObstacleNavigationMir100Sim / ObstacleAvoidanceMir100Sim

env.reset()
# add wrapper for automatic exception handlingz
env = ExceptionHandling(env)

# choose and run appropriate algorithm provided by stable-baselines
model = PPO("MlpPolicy", env, n_steps=2048, learning_rate=3e-3, verbose=1, tensorboard_log="./logs")


TIMESTEPS = 1024
err_count = 0
i = 1
while i < 4000:
    try:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="e2e_rl_ppo_rtfactor_1")
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
        if i == 1:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
        else:
            print("Loading model ", TIMESTEPS * (i - 1))
            model_path = f"{models_dir}/{TIMESTEPS*(i-1)}"
            model = PPO.load(model_path, env=env)

        continue


env.close()
