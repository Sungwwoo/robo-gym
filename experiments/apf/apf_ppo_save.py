import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO
import os
from datetime import datetime

# specify the ip of the machine running the robot-server
target_machine_ip = "163.180.177.101"  # or other xxx.xxx.xxx.xxx

models_dir = "models/basic_apf_PPO"

date = datetime.now()
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# initialize environment
env = gym.make("Basic_APF_Jackal_Kinova_Sim-v0", ip=target_machine_ip)

env.reset()
# add wrapper for automatic exception handlingz
env = ExceptionHandling(env)

# load learned model
# models_dir = "models/husky_nav3_PPO"
# model_path = f"{models_dir}/60000.zip"
# model = PPO.load(model_path, env=env)

# choose and run appropriate algorithm provided by stable-baselines
model = PPO("MlpPolicy", env, learning_rate=4e-4, n_steps=512, verbose=1, tensorboard_log="./logs/" + date.strftime("%Y%m%d-%H%M%S"))

TIMESTEPS = 2048
err_count = 0
i = 1
while i < 4000:
    try:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="basic_apf_ppo_" + date.strftime("%Y%m%d-%H%M%S"))
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
