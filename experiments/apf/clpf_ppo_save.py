import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO
from torch import nn
import os

# specify the ip of the machine running the robot-server
target_machine_ip = "163.180.177.101"  # or other xxx.xxx.xxx.xxx

run_name = "clpf_ppo_1"
models_dir = "models/" + run_name
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# initialize environment
env = gym.make("Clustered_APF_Jackal_Kinova_Sim-v0", ip=target_machine_ip)

env.reset()
# add wrapper for automatic exception handlingz
env = ExceptionHandling(env)


# choose and run appropriate algorithm provided by stable-baselines
policy_kwarg = dict(activation_fn=nn.Tanh, net_arch=[512, dict(pi=[512, 512], vf=[512, 512])])
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwarg,
    learning_rate=6e-5,
    verbose=1,
    tensorboard_log="./logs/",
)


TIMESTEPS = 1000
err_count = 0
i = 1
while i < 1000:
    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=run_name,
        )
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
        env = gym.make("Clustered_APF_Jackal_Kinova_Sim-v0", ip=target_machine_ip)
        env.reset()
        env = ExceptionHandling(env)

        del model
        print("Loading model ", TIMESTEPS * (i - 1))
        model_path = f"{models_dir}/{TIMESTEPS*(i-1)}"
        model = PPO.load(model_path, env=env)

        continue


env.close()
