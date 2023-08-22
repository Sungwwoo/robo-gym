import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import SAC
import os

# specify the ip of the machine running the robot-server
target_machine_ip = "163.180.177.101"  # or other xxx.xxx.xxx.xxx

models_dir = "models/basic_apf_SAC"
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
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs")

TIMESTEPS = 1000

for i in range(1, 4000):
    for attempt in range(0, 10):
        try:
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="basic_apf_sac")
            model.save(f"{models_dir}/{TIMESTEPS*i}")
            model.save_replay_buffer(f"{models_dir}/buffer")
        except KeyboardInterrupt as e:
            print(e)
            del env
            exit()
        except:
            print("Got an error while excueting learn(). Retrying...")
            env.close()
            del env
            env = gym.make("Basic_APF_Jackal_Kinova_Sim-v0", ip=target_machine_ip)
            env.reset()
            env = ExceptionHandling(env)

            del model
            if i == 1:
                model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
            else:
                print("Loading model ", TIMESTEPS * (i - 1))
                model_path = f"{models_dir}/{TIMESTEPS*(i-1)}.zip"
                model = SAC.load(model_path, env=env)
                model.load_replay_buffer(f"{models_dir}/buffer")
            continue

        break

env.close()
