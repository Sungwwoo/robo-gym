import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO
import torch.nn as nn
import os
from datetime import datetime

# specify the ip of the machine running the robot-server
target_machine_ip = "163.180.177.101"  # or other xxx.xxx.xxx.xxx

run_name = "local_minima"
models_dir = "models/" + run_name

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

model_path = f"{models_dir}/999000"

model = PPO.load(model_path, env=env)


episodes = 50
success = 0
sum_time = 0
best_time = 1000
for i in range(episodes):
    print("==========================================")
    print("Episode {} of {}".format(i + 1, episodes))
    obs = env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs)
        obs, reward, done, info = env.step(action)
    print(info["final_status"])
    if info["final_status"] == "success":
        sum_time += info["elapsed_time"]
        if best_time > info["elapsed_time"]:
            best_time = info["elapsed_time"]
        success += 1

print("Success Rate: {}%".format(success / episodes * 100))
print("Average Time: {:.3f} sec".format((sum_time / success)))
print("Best Time: {:.3f} sec".format(best_time))
env.close()

env.close()
