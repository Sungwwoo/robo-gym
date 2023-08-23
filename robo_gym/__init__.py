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


# Husky_ur3
register(
    id="NoObstacleNavigationHusky_ur3_Sim-v0",
    entry_point="robo_gym.envs:NoObstacleNavigationHusky_ur3_Sim",
)

register(
    id="NoObstacleNavigationHusky_ur3_Rob-v0",
    entry_point="robo_gym.envs:NoObstacleNavigationHusky_ur3_Rob",
)

register(
    id="ObstacleAvoidanceHusky_ur3_Sim-v0",
    entry_point="robo_gym.envs:ObstacleAvoidanceHusky_ur3_Sim",
)

register(
    id="ObstacleAvoidanceHusky_ur3_Rob-v0",
    entry_point="robo_gym.envs:ObstacleAvoidanceHusky_ur3_Rob",
)


# Jackal_Kinova
register(
    id="No_Obstacle_Avoidance_Jackal_Kinova_Sim-v0",
    entry_point="robo_gym.envs:No_Obstacle_Avoidance_Jackal_Kinova_Sim",
)

register(
    id="No_Obstacle_Avoidance_Jackal_Kinova_Rob-v0",
    entry_point="robo_gym.envs:No_Obstacle_Avoidance_Jackal_Kinova_Rob",
)
register(
    id="Obstacle_Avoidance_Jackal_Kinova_Sim-v0",
    entry_point="robo_gym.envs:Obstacle_Avoidance_Jackal_Kinova_Sim",
)

register(
    id="Obstacle_Avoidance_Jackal_Kinova_Rob-v0",
    entry_point="robo_gym.envs:Obstacle_Avoidance_Jackal_Kinova_Rob",
)


# Artificial Potential Field RL
register(
    id="Basic_APF_Jackal_Kinova_Sim-v0",
    entry_point="robo_gym.envs:Basic_APF_Jackal_Kinova_Sim",
)

register(
    id="Basic_APF_with_PD_Jackal_Kinova_Sim-v0",
    entry_point="robo_gym.envs:Basic_APF_with_PD_Jackal_Kinova_Sim",
)

register(
    id="Clustered_APF_Jackal_Kinova_Sim-v0",
    entry_point="robo_gym.envs:Clustered_APF_Jackal_Kinova_Sim",
)

register(
    id="Clustered_APF_with_PD_Jackal_Kinova_Sim-v0",
    entry_point="robo_gym.envs:Clustered_APF_with_PD_Jackal_Kinova_Sim",
)
