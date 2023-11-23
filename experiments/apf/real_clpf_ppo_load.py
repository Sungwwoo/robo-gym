import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO
import torch.nn as nn
import os
from datetime import datetime

# specify the ip of the machine running the robot-server
target_machine_ip = "192.168.0.81:50050"  # or other xxx.xxx.xxx.xxx

run_name = "clpf_ppo_6"
models_dir = "models/" + run_name


# initialize environment
env = gym.make("Clustered_APF_Jackal_Kinova_Rob-v0", rs_address=target_machine_ip)

env.reset()
# add wrapper for automatic exception handlingz
env = ExceptionHandling(env)

model_path = f"{models_dir}/1228000"

model = PPO.load(model_path, env=env)


episodes = 1
success = 0

sum_time = 0
sum_path_length = 0
sum_total_acc_lin = 0
sum_total_acc_ang = 0

best_time = 1000
best_path_length = 1000


for i in range(episodes):
    path_length = 0
    total_time = 0
    total_acc_lin = 0
    total_acc_ang = 0

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
        if best_time > info["elapsed_time"] > 0.0:
            best_time = info["elapsed_time"]

        sum_path_length += info["distance_moved"]
        sum_total_acc_lin += info["acc_lin"]
        sum_total_acc_ang += info["acc_ang"]
        if best_path_length > info["distance_moved"] > 0.0:
            best_path_length = info["distance_moved"]

        success += 1

print("Success Rate: {}%".format(success / episodes * 100))
print("Average Time: {:.3f} sec".format((sum_time / success)))
print("Best Time: {:.3f} sec".format(best_time))
