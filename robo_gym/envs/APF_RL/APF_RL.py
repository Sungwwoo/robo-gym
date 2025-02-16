#!/usr/bin/env python3

import sys, time, math, copy
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from robo_gym.utils import utils, jackal_kinova_utils, apf_env_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2


NUM_OBSTACLES = 3
DIST_BTW_OBSTACLES = 1.7
# State Indecies

# rs_state:
# target = [0.0] * 3
# apf_weights = [0.0] * 2
# robot_pose = [0.0] * 3
# robot_twist = [0.0] * 2
# forces = [0.0] * 3
# collision = False
# obstacles = [0.0] * 21
RS_TARGET = 0
RS_WEIGHTS = RS_TARGET + 3
RS_SCAN = RS_WEIGHTS + 2
RS_ROBOT_POSE = RS_SCAN + 811  # Laser scan length of jackal
RS_ROBOT_TWIST = RS_ROBOT_POSE + 3
RS_FORCES = RS_ROBOT_TWIST + 2
RS_COLLISION = RS_FORCES + 9
RS_ROSTIME = RS_COLLISION + 1
RS_PDGAINS = RS_ROSTIME + 1

start_points = [
    [14.0, 15.0, -np.pi],
    [7.5, 7.0, np.pi / 2],
    [7.5, 3.0, 0],
    [7.5, -3.0, 0],
    [6.0, 1.75, -np.pi],
    [3.0, 16.7, -np.pi / 2],
    [-6.0, 1.5, np.pi / 2],
    [8.0, -9.0, 0],
    [8.0, -15.0, 0],
    [6.0, -1.75, -np.pi],
    [3.0, -4.5, -np.pi / 2],
    [-6.0, -1.5, -np.pi],
]

target_points = [
    [8.0, 15.0, 0],
    [14.0, 7.0, 0],
    [14.0, 5.0, 0],
    [11.0, -3.0, 0],
    [-2.75, 17.0, 0],
    [3.0, 4.5, 0],
    [-14.0, 17.0, 0],
    [14.0, -9.0, 0],
    [14.0, -16.0, 0],
    [-2.75, -17.0, -np.pi],
    [3.0, -16.7, 0],
    [-14.0, -17.0, -np.pi],
]

e_lengths = [300, 300, 300, 300, 600, 500, 800, 300, 300, 600, 500, 800]


class Basic_APF_Jackal_Kinova(gym.Env):
    """Mobile Industrial Robots jackal_kinova base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.

    Attributes:
        jackal_kinova (:obj:): Robot utilities object.
        observation_space (:obj:): Environment observation space.
        action_space (:obj:): Environment action space.
        distance_threshold (float): Minimum distance (m) from target to consider it reached.
        min_target_dist (float): Minimum initial distance (m) between robot and target.
        max_vel (numpy.array): # Maximum allowed linear (m/s) and angular (rad/s) velocity.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.
        laser_len (int): Length of laser data array included in the environment state.
    """

    real_robot = False
    laser_len = 811
    laser_downsample_len = 32
    max_episode_steps = 150

    def __init__(self, rs_address=None, **kwargs):
        self.jackal_kinova = jackal_kinova_utils.Jackal_Kinova()
        self.apf_util = apf_env_utils.APF()

        # Using normalized action space
        # # KP, ETA
        # self.action_low = [1, 1, -10]
        # self.action_high = [10, 10, 10]

        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()

        self.initialize()
        # self.action_space = spaces.Box(
        #     low=np.array(self.action_low),
        #     high=np.array(self.action_high),
        #     dtype=np.float32,
        # )
        self.action_space = spaces.MultiDiscrete([10, 10, 7])
        self.seed()
        self.distance_threshold = 0.3
        # Maximum linear velocity (m/s) of Robot
        max_lin_vel = self.apf_util.get_max_lin_vel()
        # Maximum angular velocity (rad/s) of Robot
        max_ang_vel = self.apf_util.get_max_ang_vel()
        self.max_vel = np.array([max_lin_vel, max_ang_vel])

        self.prev_lin_vel = 0.0
        self.prev_ang_vel = 0.0
        self.episode_start_time = 0.0
        self.prev_rostime = 0.0

        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def initialize(self):
        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, start_pose=None, target_pose=None):
        """Environment reset.

        Args:
            start_pose (list[3] or np.array[3]): [x,y,yaw] initial robot position.
            target_pose (list[3] or np.array[3]): [x,y,yaw] target robot position.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.prev_base_reward = None

        self.prev_lin_vel = 0.0
        self.prev_ang_vel = 0.0
        self.episode_start_time = 0.0
        self.prev_rostime = 0.0

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        env_num = int(np.random.randint(0, len(start_points)))

        # Set Robot starting position
        if start_pose:
            assert len(start_pose) == 3
        else:
            start_pose = self._get_start_pose(env_num)

        rs_state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose) == 3
        else:
            target_pose = self._get_target(env_num)
        rs_state[RS_TARGET : RS_TARGET + 3] = target_pose

        self.max_episode_steps = e_lengths[env_num] * 2

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state=rs_state.tolist())
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state) == self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        # target pose: cartesian to polar coordinate system
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        return self.state

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}
        r = 0
        # Reward base: Distance to target

        # Calculate distance to the target
        target_coords = np.array([rs_state[RS_TARGET], rs_state[RS_TARGET + 1]])
        robot_coords = np.array([rs_state[RS_ROBOT_POSE], rs_state[RS_ROBOT_POSE + 1]])
        euclidean_dist_2d = np.linalg.norm(target_coords - robot_coords, axis=-1)

        polar_r, att_theta = utils.cartesian_to_polar_2d(
            x_target=0,
            y_target=0,
            x_origin=rs_state[RS_FORCES],
            y_origin=rs_state[RS_FORCES + 1],
        )

        polar_r, rep_theta = utils.cartesian_to_polar_2d(
            x_target=0,
            y_target=0,
            x_origin=rs_state[RS_FORCES + 3],
            y_origin=rs_state[RS_FORCES + 4],
        )

        base_reward = -30.0 * euclidean_dist_2d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        # Negative rewards
        if abs(att_theta - rep_theta) < np.pi / 6:
            if action[2] == 0.0:
                reward -= 2.0
        else:
            if action[2] != 0.0:
                reward -= 1.0

        if self.prev_rostime == 0.0:
            self.prev_rostime = rs_state[RS_ROSTIME]
            self.episode_start_time = rs_state[RS_ROSTIME]
        else:
            # High acceleration
            timediff = rs_state[RS_ROSTIME] - self.prev_rostime

            # 1: Continous Penalty
            # r = 10 * (abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) + abs(rs_state[RS_ROBOT_TWIST] - self.prev_ang_vel))

            # 2: Discrete Penalty
            # if abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) / timediff > self.apf_util.get_max_lin_acc():
            #     r = 5
            # if abs(rs_state[RS_ROBOT_TWIST + 1] - self.prev_ang_vel) / timediff > self.apf_util.get_max_ang_acc():
            #     r = 5

            # reward -= r
            self.prev_rostime = rs_state[RS_ROSTIME]

        self.prev_lin_vel = rs_state[RS_ROBOT_TWIST]
        self.prev_ang_vel = rs_state[RS_ROBOT_TWIST + 1]

        # path length (episode length)
        reward -= 60 / self.max_episode_steps

        if not self.real_robot:
            # End episode if robot is collides with an object.
            if self._sim_robot_collision(rs_state):
                reward = -60.0
                done = True
                info["final_status"] = "collision"
                print("collision occured")
                print("Episode Length: ", str(self.elapsed_steps))
                print()

            if self._min_laser_reading_below_threshold(rs_state):
                reward = -60.0
                done = True
                info["final_status"] = "front collision"
                print("front blocked")
                print("Episode Length: ", str(self.elapsed_steps))
                print()

            # if self._robot_outside_of_boundary_box(rs_state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3]):
            #     reward = -30.0
            #     done = True
            #     info["final_status"] = "out of boundary"
            #     print("Robot out of boundary")
            #     print()

        # Target Reached
        if euclidean_dist_2d < self.distance_threshold:
            reward = 30.0
            done = True
            info["final_status"] = "success"
            info["elapsed_time"] = rs_state[RS_ROSTIME] - self.episode_start_time
            print("Target Reached!")
            print("Episode Length: ", str(self.elapsed_steps))
            print()

        # Time step exceeded
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info["final_status"] = "max_steps_exceeded"
            print("max_step_exceeded")
            print("Distance from target: ", str(euclidean_dist_2d))
            print()

        return reward, done, info

    def step(self, action):
        """Send action, get rs_state, and calculate reward"""
        action = action.astype(np.int32)

        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        # Convert environment action to Robot Server action
        rs_action = copy.deepcopy(action)

        # Multiply normalized actions
        rs_action = self._denormalize_actions(rs_action)

        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        # Get states from robot_server.RosBridge.get_state()
        rs_state = self.client.get_state_msg().state

        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # Assign reward
        reward, done, info = self._reward(rs_state=rs_state, action=rs_action)

        return self.state, reward, done, info

    def render(self):
        pass

    def _denormalize_actions(self, action):
        rs_action = np.zeros(3, dtype=np.float32)
        rs_action[0] = 20.0 * (action[0] + 1)
        rs_action[1] = 50.0 * (action[1] + 1)
        rs_action[2] = -np.pi + (action[2] * np.pi / 3)
        return rs_action

    def _get_robot_server_state_len(self):
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """

        target = [0.0] * 3
        apf_weights = [0.0] * 2
        scan = [0.0] * self.laser_len
        robot_pose = [0.0] * 3
        robot_twist = [0.0] * 2
        forces = [0.0] * 9
        collision = [False]
        rostime = [0.0]

        rs_state = (
            target + apf_weights + scan + robot_pose + robot_twist + forces + collision + rostime
        )

        return len(rs_state)

    def _get_env_state_len(self):
        """Get length of the environment state.

        Describes the composition of the environment state and returns
        its length.

        Returns:
            int: Length of the environment state

        """

        target_polar_coordinates = [0.0] * 2
        scan = [0.0] * self.laser_downsample_len
        weights = [0.0] * 2
        forces = [0.0] * 9
        robot_twist = [0.0] * 2  # Linear, Angular
        env_state = target_polar_coordinates + scan + weights + forces + robot_twist

        return len(env_state)

    def _get_start_pose(self, env_num):
        """Get initial robot coordinates.

        For the real robot the initial coordinates are its current coordinates
        whereas for the simulated robot the initial coordinates are
        randomly generated.

        Returns:
            numpy.array: [x,y,yaw] robot initial coordinates.
        """

        return start_points[env_num]

    def _get_target(self, env_num):
        """Generate coordinates of the target at a minimum distance from the robot.

        Args:
            robot_coordinates (list): [x,y,yaw] coordinates of the robot.

        Returns:
            numpy.array: [x,y,yaw] coordinates of the target.

        """

        return target_points[env_num]

    def _robot_server_state_to_env_state(self, rs_state):
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """

        # Convert to numpy array and remove NaN values
        rs_state = np.nan_to_num(np.array(rs_state))

        # Distance and direction from robot to target (for reward calculation?)
        # Transform cartesian coordinates of target to polar coordinates
        polar_r, polar_theta = utils.cartesian_to_polar_2d(
            x_target=rs_state[RS_TARGET],
            y_target=rs_state[RS_TARGET + 1],
            x_origin=rs_state[RS_ROBOT_POSE],
            y_origin=rs_state[RS_ROBOT_POSE + 1],
        )
        # Rotate origin of polar coordinates frame to be matching with robot frame and normalize to +/- pi
        polar_theta = utils.normalize_angle_rad(polar_theta - rs_state[RS_ROBOT_POSE + 2])

        laser = utils.downsample_list_to_len(
            rs_state[RS_SCAN : RS_SCAN + self.laser_len], self.laser_downsample_len
        )

        state = np.concatenate(
            (
                np.array([polar_r, polar_theta]),
                laser,
                rs_state[RS_WEIGHTS : RS_WEIGHTS + 2],
                rs_state[RS_FORCES : RS_FORCES + 9],
                rs_state[RS_ROBOT_TWIST : RS_ROBOT_TWIST + 2],
            )
        )

        return state.astype(np.float32)

    def _get_observation_space(self):
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """

        # Target coordinates range
        max_target_coords = np.array([np.inf, np.pi])
        min_target_coords = np.array([-np.inf, -np.pi])

        # Robot velocity range tolerance
        vel_tolerance = 2.0

        # Robot velocity range used to determine if there is an error in the sensor readings
        max_lin_vel = self.apf_util.get_max_lin_vel() + vel_tolerance
        min_lin_vel = self.apf_util.get_min_lin_vel() - vel_tolerance
        max_ang_vel = self.apf_util.get_max_ang_vel() + vel_tolerance
        min_ang_vel = -self.apf_util.get_max_ang_vel() - vel_tolerance

        # LaserScan
        max_laser = np.full(self.laser_downsample_len, np.inf)  # Using inf due to sensor noise
        min_laser = np.full(self.laser_downsample_len, 0.0)

        # Weights
        max_weights = np.array([np.inf, np.inf])
        min_weights = np.array([0, 0])

        # Attractive, Repulsive, Total Forces
        max_forces = np.array([np.inf for i in range(0, 9)])
        min_forces = np.array([-np.inf for i in range(0, 9)])

        # Velocities
        max_vel = np.array([max_lin_vel, max_ang_vel])
        min_vel = np.array([min_lin_vel, min_ang_vel])

        # Definition of environment observation_space
        obs_space_max = np.concatenate(
            (max_target_coords, max_laser, max_weights, max_forces, max_vel)
        )
        obs_space_min = np.concatenate(
            (min_target_coords, min_laser, min_weights, min_forces, min_vel)
        )

        return spaces.Box(low=obs_space_min, high=obs_space_max, dtype=np.float32)

    def _robot_outside_of_boundary_box(self, robot_coordinates):
        """Check if robot is outside of boundary box.

        Check if the robot is outside of the boundaries defined as a box with
        its center in the origin of the map and sizes width and height.

        Args:
            robot_coordinates (list): [x,y] Cartesian coordinates of the robot.

        Returns:
            bool: True if outside of boundaries.

        """

        # Dimensions of boundary box in m, the box center corresponds to the map origin
        x = 9
        y = 6

        if (
            robot_coordinates[0] < -1
            or robot_coordinates[0] > x
            or np.absolute(robot_coordinates[1]) > (y / 2)
        ):
            return True
        else:
            return False

    def _sim_robot_collision(self, rs_state):
        """Get status of simulated collision sensor.

        Used only for simulated Robot Server.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if the robot is in collision.

        """
        ret = False
        if rs_state[RS_COLLISION] == 1:
            ret = True
        else:
            ret = False

        return ret

    def _min_laser_reading_below_threshold(self, rs_state):
        """Check if any of the laser readings is below a threshold.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if any of the laser readings is below the threshold.

        """

        threshold = 0.3
        if min(rs_state[RS_SCAN : RS_SCAN + self.laser_len]) < threshold:
            return True
        else:
            return False

    def _robot_close_to_sim_obstacle(self, rs_state):
        """Check if the robot is too close to one of the obstacles in simulation.

        Check if one of the corner of the robot's base has a distance shorter
        than the safety radius from one of the simulated obstacles. Used only for
        simulated Robot Server.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if the robot is too close to an obstacle.

        """

        # Minimum distance from obstacle center
        safety_radius = 0.55

        robot_close_to_obstacle = False
        robot_corners = self.jackal_kinova.get_corners_positions(
            rs_state[RS_ROBOT_POSE],
            rs_state[RS_ROBOT_POSE + 1],
            rs_state[RS_ROBOT_POSE + 2],
        )

        for corner in robot_corners:
            for obstacle_coord in self.sim_obstacles:
                if utils.point_inside_circle(
                    corner[0],
                    corner[1],
                    obstacle_coord[0],
                    obstacle_coord[1],
                    safety_radius,
                ):
                    robot_close_to_obstacle = True

        return robot_close_to_obstacle

    def _generate_obstacles_positions(self, rs_state):
        """Generate random positions for n obstacles.

        Used only for simulated Robot Server.

        """

        self.sim_obstacles = []
        i = 0
        while i < NUM_OBSTACLES:
            pose = self._generate_obsetacle_pos()
            count = 0
            while not self._is_valid_obstacle(pose, rs_state):
                # Regenerate if valid position is not found for 100 times
                if count > 100:
                    i -= 1
                    self.sim_obstacles.pop()
                pose = self._generate_obsetacle_pos()
                count += 1
            self.sim_obstacles.append(pose)
            i += 1

    def _generate_obsetacle_pos(self):
        x = self.np_random.uniform(low=3.0, high=8.0)
        y = self.np_random.uniform(low=-2.1, high=2.1)
        yaw = self.np_random.uniform(low=-np.pi, high=np.pi)
        return [x, y, yaw]

    def _is_valid_obstacle(self, obst_pose, rs_state):
        if (
            np.sqrt(
                (obst_pose[0] - rs_state[RS_TARGET]) ** 2
                + (obst_pose[1] - rs_state[RS_TARGET + 1]) ** 2
            )
            < 0.8
        ):
            return False
        for i in range(0, len(self.sim_obstacles)):
            if (
                np.sqrt(
                    (self.sim_obstacles[i][0] - obst_pose[0]) ** 2
                    + (self.sim_obstacles[i][1] - obst_pose[1]) ** 2
                )
                < DIST_BTW_OBSTACLES
            ):
                return False
        return True


class Basic_APF_Jackal_Kinova_Sim(Basic_APF_Jackal_Kinova, Simulation):
    cmd = "roslaunch apf_robot_server sim_robot_server.launch world_name:=learning_world_4.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=True, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Basic_APF_Jackal_Kinova.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class Basic_APF_Jackal_Kinova_Rob(Basic_APF_Jackal_Kinova):
    real_robot = True


class Basic_APF_with_PD_Jackal_Kinova(Basic_APF_Jackal_Kinova):
    def initialize(self):
        # normalized linear_kd, angular_kp, angular_kd
        self.action_low.extend([0, 0, 0])
        self.action_high.extend([0.9, 1, 1])
        return

    def _denormalize_actions(self, action):
        rs_action = np.zeros(5, dtype=np.float32)
        rs_action[0] = 100 * action[0]
        rs_action[1] = 500 * action[1]
        rs_action[2] = action[2]
        rs_action[3] = 10 * action[3]
        rs_action[4] = 10 * action[4]
        return rs_action

    def _get_robot_server_state_len(self):
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """

        target = [0.0] * 3
        apf_weights = [0.0] * 2
        scan = [0.0] * self.laser_len
        robot_pose = [0.0] * 3
        robot_twist = [0.0] * 2
        forces = [0.0] * 9
        collision = False
        rostime = [0.0]
        pd_gains = [0.0] * 3

        rs_state = (
            target
            + apf_weights
            + scan
            + robot_pose
            + robot_twist
            + forces
            + [collision]
            + rostime
            + pd_gains
        )

        return len(rs_state)


class Basic_APF_with_PD_Jackal_Kinova_Sim(Basic_APF_with_PD_Jackal_Kinova, Simulation):
    cmd = "roslaunch apf_robot_server sim_robot_server_with_pd.launch world_name:=learning_world_4.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=True, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Basic_APF_Jackal_Kinova.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class Basic_APF_with_PD_Jackal_Kinova_Rob(Basic_APF_with_PD_Jackal_Kinova):
    real_robot = True
