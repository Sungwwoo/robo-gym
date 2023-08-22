import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO
import os

# specify the ip of the machine running the robot-server
target_machine_ip = "163.180.177.101"  # or other xxx.xxx.xxx.xxx

models_dir = "models/clustered_apf_PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# initialize environment
env = gym.make("Clustered_APF_Jackal_Kinova_Sim-v0", ip=target_machine_ip)
# env NoObstacleNavigationMir100Sim / ObstacleAvoidanceMir100Sim

env.reset()
# add wrapper for automatic exception handlingz
env = ExceptionHandling(env)

# load learned model
# models_dir = "models/husky_nav3_PPO"
# model_path = f"{models_dir}/60000.zip"
# model = PPO.load(model_path, env=env)

model_path = f"{models_dir}/78000.zip"

model = PPO.load(model_path, env=env)

print("\nAction Space Info:")
print(env.action_space)
print()
# print("Observation Space Info:")
# print(env.observation_space)

TIMESTEPS = 1000

TIMESTEPS = 1000

for i in range(1, 4000):
    for attempt in range(0, 10):
        try:
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="clustered_apf_ppo")
            model.save(f"{models_dir}/{TIMESTEPS*i}")
        except KeyboardInterrupt as e:
            print(e)
            exit()
        except:
            print("Got an error while excueting learn(). Retrying...")
            env.close()
            del env
            env = gym.make("Clustered_APF_Jackal_Kinova_Sim-v0", ip=target_machine_ip)
            env.reset()
            env = ExceptionHandling(env)
            del model
            if i == 1:
                model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
            else:
                print("Loading model ", TIMESTEPS * (i - 1))
                model_path = f"{models_dir}/{TIMESTEPS*(i-1)}.zip"
                model = PPO.load(model_path, env=env)
            continue
        break
env.close()
