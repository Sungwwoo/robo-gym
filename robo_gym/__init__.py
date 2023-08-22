from gym.envs.registration import register

# naming convention: EnvnameRobotSim

# Example Environments
register(
    id="ExampleEnvSim-v0",
    entry_point="robo_gym.envs:ExampleEnvSim",
)

register(
    id="ExampleEnvRob-v0",
    entry_point="robo_gym.envs:ExampleEnvRob",
)
