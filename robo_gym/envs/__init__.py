# Example
from robo_gym.envs.example.example_env import ExampleEnvSim, ExampleEnvRob


# Husky_ur3
from robo_gym.envs.husky_ur3.husky_ur3 import (
    NoObstacleNavigationHusky_ur3_Sim,
    NoObstacleNavigationHusky_ur3_Rob,
)
from robo_gym.envs.husky_ur3.husky_ur3 import (
    ObstacleAvoidanceHusky_ur3_Sim,
    ObstacleAvoidanceHusky_ur3_Rob,
)


# jackal_kinova
from robo_gym.envs.jackal_kinova.jackal_kinova import (
    No_Obstacle_Avoidance_Jackal_Kinova_Sim,
    No_Obstacle_Avoidance_Jackal_Kinova_Rob,
    Obstacle_Avoidance_Jackal_Kinova_Sim,
    Obstacle_Avoidance_Jackal_Kinova_Rob,
)

# Artificial Potential Field RL
from robo_gym.envs.APF_RL.APF_RL import (
    Basic_APF_Jackal_Kinova_Sim,
    Basic_APF_Jackal_Kinova_Rob,
)
from robo_gym.envs.APF_RL.APF_RL import (
    Basic_APF_with_PD_Jackal_Kinova_Sim,
    Basic_APF_with_PD_Jackal_Kinova_Rob,
)
from robo_gym.envs.APF_RL.CLPF_RL import (
    Clustered_APF_Jackal_Kinova_Sim,
    Clustered_APF_Jackal_Kinova_Rob,
)
from robo_gym.envs.APF_RL.CLPF_RL import (
    Clustered_APF_with_PD_Jackal_Kinova_Sim,
    Clustered_APF_with_PD_Jackal_Kinova_Rob,
)
