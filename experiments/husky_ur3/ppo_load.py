import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO
import os

# specify the ip of the machine running the robot-server
target_machine_ip = '127.0.0.1' # or other xxx.xxx.xxx.xxx

env = gym.make('ObstacleAvoidanceHusky_ur3_Sim-v0', ip=target_machine_ip, gui=True)
env.reset()

models_dir = "models/husky_obs2_PPO"
model_path = f"{models_dir}/247000.zip"

model = PPO.load(model_path, env=env)

episodes = 20

for i in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _state= model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()