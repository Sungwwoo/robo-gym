import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO

import os

# specify the ip of the machine running the robot-server
target_machine_ip = "163.180.177.101"  # or other xxx.xxx.xxx.xxx

env = gym.make("Obstacle_Avoidance_Jackal_Kinova_Sim-v0", ip=target_machine_ip, gui=True)
env.reset()

models_dir = "models/e2e_rl_PPO_11"
model_path = f"{models_dir}/601000"

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

print("Success Rate: {:.3f}%".format(success / episodes * 100))
print("Average Time: {:.3f} sec".format((sum_time / success)))
print("Best Time: {:.3f} sec".format(best_time))
env.close()
