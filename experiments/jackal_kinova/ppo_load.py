import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO
import os

# specify the ip of the machine running the robot-server
target_machine_ip = "192.168.0.32"  # or other xxx.xxx.xxx.xxx

env = gym.make(
    "No_Obstacle_Avoidance_Jackal_Kinova_Sim-v0", ip=target_machine_ip, gui=True
)
env.reset()

models_dir = "models/no_obst_e2e_rl_PPO_2"
model_path = f"{models_dir}/165000"

model = PPO.load(model_path, env=env)

episodes = 20

for i in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()
