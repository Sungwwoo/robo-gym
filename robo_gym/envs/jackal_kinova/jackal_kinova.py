#!/usr/bin/env python3

import sys, time, math, copy
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from robo_gym.utils import utils, jackal_kinova_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2


NUM_OBSTACLES = 3
DIST_BTW_OBSTACLES = 1.8
# self.target = [0.0] * 3
# self.base_pose = [0.0] * 3
# self.base_twist = [0.0] * 2
# self.scan = [0.0] * self.laser_len
# self.collision = False
# self.obstacle_0 = [0.0] * 3
# self.obstacle_1 = [0.0] * 3
# self.obstacle_2 = [0.0] * 3
# self.obstacle_3 = [0.0] * 3
# self.obstacle_4 = [0.0] * 3
# self.obstacle_5 = [0.0] * 3
# self.obstacle_6 = [0.0] * 3

RS_TARGET = 0
RS_ROBOT_POSE = RS_TARGET + 3
RS_ROBOT_TWIST = RS_ROBOT_POSE + 3
RS_SCAN = RS_ROBOT_TWIST + 2
RS_COLLISION = RS_SCAN + 811  # Raser Scan Length
RS_ROSTIME = RS_COLLISION + 1
RS_OBSTACLES = RS_ROSTIME + 1

start_points = [
    [14.0, 15.0, -np.pi],
    [8.0, 7.0, np.pi / 2],
    [8.0, 3.0, 0],
    [8.0, -3.0, 0],
    [6.0, 1.75, -np.pi],
    [3.0, 16.7, -np.pi / 2],
    [-6.0, 1.5, 0],
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

e_lengths = [150, 150, 150, 150, 400, 300, 500, 150, 150, 400, 300, 500]


class No_Obstacle_Avoidance_Jackal_Kinova(gym.Env):
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
    downsample_len = 32
    max_episode_steps = 150

    def __init__(self, rs_address=None, **kwargs):
        self.jackal_kinova = jackal_kinova_utils.Jackal_Kinova()

        action_low = [0.0, self.jackal_kinova.get_min_ang_vel()]  # linear, angular
        action_high = [
            self.jackal_kinova.get_max_lin_vel(),
            self.jackal_kinova.get_max_ang_vel(),
        ]  # linear, angular

        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()
        self.action_space = spaces.Box(low=np.array(action_low), high=np.array(action_high), dtype=np.float32)
        self.seed()
        self.distance_threshold = 0.6
        self.min_target_dist = 1.0
        # Maximum linear velocity (m/s) of Husky
        max_lin_vel = self.jackal_kinova.get_max_lin_vel()
        # Maximum angular velocity (rad/s) of Husky
        max_ang_vel = self.jackal_kinova.get_max_ang_vel()
        self.max_vel = np.array([max_lin_vel, max_ang_vel])

        self.episode_start_time = 0.0
        self.acc_penalty = 0.0
        self.zero_vel_penalty = 0.0
        self.prev_lin_vel = 0.0
        self.prev_ang_vel = 0.0
        self.prev_rostime = 0.0

        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

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
        self.acc_penalty = 0.0
        self.zero_vel_penalty = 0.0

        self.prev_lin_vel, self.prev_ang_vel = 0.0, 0.0
        self.episode_start_time = 0.0
        self.prev_rostime = 0.0

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set Robot starting position
        if start_pose:
            assert len(start_pose) == 3
        else:
            start_pose = self._get_start_pose()

        rs_state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose) == 3
        else:
            target_pose = self._get_target(start_pose)
        rs_state[RS_TARGET : RS_TARGET + 3] = target_pose

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
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        return self.state

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}

        # Reward base: Distance to target

        # Calculate distance to the target
        target_coords = np.array([rs_state[RS_TARGET], rs_state[RS_TARGET + 1]])
        robot_coords = np.array([rs_state[RS_ROBOT_POSE], rs_state[RS_ROBOT_POSE + 1]])
        euclidean_dist_2d = np.linalg.norm(target_coords - robot_coords, axis=-1)
        # print("Target coordinate: {:.3f}, {:.3f}".format(rs_state[RS_TARGET], rs_state[RS_TARGET + 1]))
        # print("Robot coordinate: {:.3f}, {:.3f}".format(rs_state[RS_ROBOT_POSE], rs_state[RS_ROBOT_POSE + 1]))

        # self.state[0]: Euclidean distance to goal
        # self.state[1]: Heading error
        base_reward = -80 * euclidean_dist_2d

        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        if abs(self.state[1]) < np.pi / 3:
            reward += 1
        else:
            reward -= 1

        # Negative rewards

        # Power used by the motors
        linear_power = abs(action[0] * 0.15)
        angular_power = abs(action[1] * 0.3)
        reward -= linear_power
        reward -= angular_power

        # Spinning in plance
        if action[0] < 0.01 and abs(action[1]) > 0.0:
            reward -= 0.3

        if self.prev_rostime == 0.0:
            self.prev_rostime = rs_state[RS_ROSTIME]
            self.episode_start_time = rs_state[RS_ROSTIME]
        else:
            if self.acc_penalty > -200:
                # High acceleration
                # 1: Continous Penalty
                # r = 10 * (abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) + abs(rs_state[RS_ROBOT_TWIST] - self.prev_ang_vel))

                # 2: Discrete Penalty
                timediff = rs_state[RS_ROSTIME] - self.prev_rostime
                # r = 5
                # if abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) / timediff > self.apf_util.get_max_lin_acc():
                #     self.acc_penalty = self.acc_penalty - r
                # if abs(rs_state[RS_ROBOT_TWIST + 1] - self.prev_ang_vel) / timediff > self.apf_util.get_max_ang_acc():
                #     self.acc_penalty = self.acc_penalty - r

                # reward = reward - r
            self.prev_rostime = rs_state[RS_ROSTIME]

        self.prev_lin_vel = rs_state[RS_ROBOT_TWIST]
        self.prev_ang_vel = rs_state[RS_ROBOT_TWIST + 1]

        # Long path length (episode length)
        reward -= 0.7

        if not self.real_robot:
            # End episode if robot is collides with an object.
            if self._sim_robot_collision(rs_state):
                reward = -10.0
                done = True
                info["final_status"] = "collision"
                print("collision occured")
                print("Episode Length: ", str(self.elapsed_steps))

            # if self._min_laser_reading_below_threshold(rs_state):
            #     reward -= 0.3

            if self._robot_outside_of_boundary_box(rs_state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3]):
                reward = -10.0
                done = True
                info["final_status"] = "out of boundary"
                print("Robot out of boundary")

        # Target Reached
        if euclidean_dist_2d < self.distance_threshold:
            reward = 10
            done = True
            info["final_status"] = "success"
            info["elapsed_time"] = rs_state[RS_ROSTIME] - self.episode_start_time
            print("Target Reached!")
            print("Episode Length: ", str(self.elapsed_steps))

        # Time step exceeded
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info["final_status"] = "max_steps_exceeded"
            print("max_step_exceeded")
            print("Distance from target: ", str(euclidean_dist_2d))

        return reward, done, info

    def step(self, action):
        action = action.astype(np.float32)

        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        # Convert environment action to Robot Server action
        rs_action = copy.deepcopy(action)

        # Scale action (use when the normalized action space is used)
        # rs_action = np.multiply(action, self.max_vel)

        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state
        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # Assign reward
        reward, done, info = self._reward(rs_state=rs_state, action=action)

        return self.state, reward, done, info

    def render(self):
        pass

    def _get_robot_server_state_len(self):
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """

        target = [0.0] * 3
        base_pose = [0.0] * 3
        base_twist = [0.0] * 2
        scan = [0.0] * self.laser_len
        collision = False
        rostime = [0.0]

        rs_state = target + base_pose + base_twist + scan + [collision] + rostime

        return len(rs_state)

    def _get_env_state_len(self):
        """Get length of the environment state.

        Describes the composition of the environment state and returns
        its length.

        Returns:
            int: Length of the environment state

        """

        target_polar_coordinates = [0.0] * 2
        base_twist = [0.0] * 2
        laser = [0.0] * self.downsample_len
        env_state = target_polar_coordinates + base_twist + laser

        return len(env_state)

    def _get_start_pose(self):
        """Get initial robot coordinates.

        For the real robot the initial coordinates are its current coordinates
        whereas for the simulated robot the initial coordinates are
        randomly generated.

        Returns:
            numpy.array: [x,y,yaw] robot initial coordinates.

        """

        if self.real_robot:
            # Take current robot position as start position
            start_pose = self.client.get_state_msg().state[3:6]
        else:
            # Create random starting position
            x = self.np_random.uniform(low=-1.7, high=1.7)
            if np.random.choice(a=[True, False]):
                y = self.np_random.uniform(low=-2.7, high=-2.1)
            else:
                y = self.np_random.uniform(low=2.1, high=2.7)
            yaw = self.np_random.uniform(low=-np.pi / 3, high=np.pi / 3)

            x, y = 0, 0
            start_pose = [x, y, yaw]

        return start_pose

    def _get_target(self, robot_coordinates):
        """Generate coordinates of the target at a minimum distance from the robot.

        Args:
            robot_coordinates (list): [x,y,yaw] coordinates of the robot.

        Returns:
            numpy.array: [x,y,yaw] coordinates of the target.

        """

        x_t = self.np_random.uniform(low=3.0, high=7.5)
        y_t = self.np_random.uniform(low=-2.1, high=2.1)
        yaw_t = 0.0
        # target_dist = np.linalg.norm(np.array([x_t, y_t]) - np.array(robot_coordinates[0:2]), axis=-1)

        return [x_t, y_t, yaw_t]

    def _robot_server_state_to_env_state(self, rs_state):
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Convert to numpy array and remove NaN values
        rs_state = np.nan_to_num(np.array(rs_state))

        # Transform cartesian coordinates of target to polar coordinates
        polar_r, polar_theta = utils.cartesian_to_polar_2d(
            x_target=rs_state[RS_TARGET],
            y_target=rs_state[RS_TARGET + 1],
            x_origin=rs_state[RS_ROBOT_POSE],
            y_origin=rs_state[RS_ROBOT_POSE + 1],
        )
        # Rotate origin of polar coordinates frame to be matching with robot frame and normalize to +/- pi
        polar_theta = utils.normalize_angle_rad(polar_theta - rs_state[RS_ROBOT_POSE + 2])
        # Get Laser scanners data
        raw_laser_scan = rs_state[RS_SCAN : RS_SCAN + self.laser_len]

        # Downsampling of laser values by picking every n-th value
        if self.laser_len > 0:
            laser = utils.downsample_list_to_len(raw_laser_scan, self.downsample_len)
            # Compose environment state
            state = np.concatenate(
                (
                    np.array([polar_r, polar_theta]),
                    rs_state[RS_ROBOT_TWIST : RS_ROBOT_TWIST + 2],
                    laser,
                )
            )
        else:
            # Compose environment state
            state = np.concatenate(
                (
                    np.array([polar_r, polar_theta]),
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
        max_lin_vel = self.jackal_kinova.get_max_lin_vel() + vel_tolerance
        min_lin_vel = self.jackal_kinova.get_min_lin_vel() - vel_tolerance
        max_ang_vel = self.jackal_kinova.get_max_ang_vel() + vel_tolerance
        min_ang_vel = self.jackal_kinova.get_min_ang_vel() - vel_tolerance
        max_vel = np.array([max_lin_vel, max_ang_vel])
        min_vel = np.array([min_lin_vel, min_ang_vel])

        # Laser readings range
        max_laser = np.full(self.downsample_len, 25.0)
        min_laser = np.full(self.downsample_len, 0.0)

        # Definition of environment observation_space
        max_obs = np.concatenate((max_target_coords, max_vel, max_laser))
        min_obs = np.concatenate((min_target_coords, min_vel, min_laser))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _sim_robot_collision(self, rs_state):
        """Get status of simulated collision sensor.

        Used only for simulated Robot Server.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if the robot is in collision.

        """

        if rs_state[RS_COLLISION] == 1:
            return True
        else:
            return False

    def _min_laser_reading_below_threshold(self, rs_state):
        """Check if any of the laser readings is below a threshold.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if any of the laser readings is below the threshold.

        """

        threshold = 0.1
        if min(rs_state[RS_SCAN : RS_SCAN + self.laser_len]) < threshold:
            return True
        else:
            return False

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
        width = 9
        height = 6

        if robot_coordinates[0] < -2 or robot_coordinates[0] > width or np.absolute(robot_coordinates[1]) > (height / 2):
            return True
        else:
            return False


class No_Obstacle_Avoidance_Jackal_Kinova_Sim(No_Obstacle_Avoidance_Jackal_Kinova, Simulation):
    cmd = "roslaunch jackal_kinova_robot_server sim_robot_server_no_obstacle.launch world_name:=lab_6by9_no_obst.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=True, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Obstacle_Avoidance_Jackal_Kinova.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class No_Obstacle_Avoidance_Jackal_Kinova_Rob(No_Obstacle_Avoidance_Jackal_Kinova):
    real_robot = True


class Obstacle_Avoidance_Jackal_Kinova(No_Obstacle_Avoidance_Jackal_Kinova):
    """Mobile Robots jackal_kinova base environment.

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

    def step(self, action):
        action = action.astype(np.float32)

        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        # Convert environment action to Robot Server action
        rs_action = copy.deepcopy(action)

        # Scale action (use when the normalized action space is used)
        # rs_action = np.multiply(action, self.max_vel)

        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state
        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # Assign reward
        reward, done, info = self._reward(rs_state=rs_state, action=action)

        return self.state, reward, done, info

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
        self.acc_penalty = 0.0
        self.zero_vel_penalty = 0.0

        self.prev_lin_vel, self.prev_ang_vel = 0.0, 0.0
        self.episode_start_time = 0.0
        self.prev_rostime = 0.0

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set Robot starting position
        if start_pose:
            assert len(start_pose) == 3
        else:
            start_pose = self._get_start_pose()

        rs_state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose) == 3
        else:
            target_pose = self._get_target(start_pose)
        rs_state[RS_TARGET : RS_TARGET + 3] = target_pose

        # Generate obstacles positions
        self._generate_obstacles_positions(rs_state)
        for i in range(0, NUM_OBSTACLES):
            rs_state[RS_OBSTACLES + 3 * i : RS_OBSTACLES + 3 * (i + 1)] = self.sim_obstacles[i]

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
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        return self.state

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}

        # Reward base: Distance to target

        # Calculate distance to the target
        target_coords = np.array([rs_state[RS_TARGET], rs_state[RS_TARGET + 1]])
        robot_coords = np.array([rs_state[RS_ROBOT_POSE], rs_state[RS_ROBOT_POSE + 1]])
        euclidean_dist_2d = np.linalg.norm(target_coords - robot_coords, axis=-1)
        # print("Target coordinate: {:.3f}, {:.3f}".format(rs_state[RS_TARGET], rs_state[RS_TARGET + 1]))
        # print("Distance: {:.3f}".format(euclidean_dist_2d))
        # print("Robot heading: {:.3f}".format(rs_state[RS_ROBOT_POSE + 2]))
        # print("Heading Error:: {:.3f}".format(self.state[1]))

        base_reward = -30 * euclidean_dist_2d

        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        if abs(self.state[1]) < np.pi / 3:
            reward += 1.3
        else:
            reward -= 1

        # Negative rewards
        # if rs_state[RS_ROBOT_TWIST] < 0.01 and self.zero_vel_penalty > -200:
        #     self.zero_vel_penalty = self.zero_vel_penalty - 5
        #     reward = reward - 5

        # Spinning in plance
        if action[0] < 0.01 and abs(action[1]) > 0.0:
            reward -= 0.3

        # Power used by the motors
        # linear_power = abs(action[0] * 0.15)
        # angular_power = abs(action[1] * 0.3)
        # reward -= linear_power
        # reward -= angular_power

        if self.prev_rostime == 0.0:
            self.prev_rostime = rs_state[RS_ROSTIME]
            self.episode_start_time = rs_state[RS_ROSTIME]
        else:
            if self.acc_penalty > -200:
                # High acceleration
                # 1: Continous Penalty
                # r = 10 * (abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) + abs(rs_state[RS_ROBOT_TWIST] - self.prev_ang_vel))

                # 2: Discrete Penalty
                timediff = rs_state[RS_ROSTIME] - self.prev_rostime
                # r = 5
                # if abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) / timediff > self.apf_util.get_max_lin_acc():
                #     self.acc_penalty = self.acc_penalty - r
                # if abs(rs_state[RS_ROBOT_TWIST + 1] - self.prev_ang_vel) / timediff > self.apf_util.get_max_ang_acc():
                #     self.acc_penalty = self.acc_penalty - r

                # reward = reward - r
            self.prev_rostime = rs_state[RS_ROSTIME]

        self.prev_lin_vel = rs_state[RS_ROBOT_TWIST]
        self.prev_ang_vel = rs_state[RS_ROBOT_TWIST + 1]

        # Long path length (episode length)
        reward -= 0.7

        if not self.real_robot:
            # Negative reward if robot is too close to the obstacles
            if self._robot_close_to_sim_obstacle(rs_state):
                reward = -30.0
                done = True
                info["final_status"] = "robot close to obstacle"
                print("robot close to obstacle")
                print("Episode Length: ", str(self.elapsed_steps))

            # End episode if robot is collides with an object.
            if self._sim_robot_collision(rs_state):
                reward = -30.0
                done = True
                info["final_status"] = "collision"
                print("collision occured")
                print("Episode Length: ", str(self.elapsed_steps))

            # if self._min_laser_reading_below_threshold(rs_state):
            #     reward = -10.0
            #     done = True
            #     info["final_status"] = "front collision"
            #     print("front blocked")
            #     print("Episode Length: ", str(self.elapsed_steps))

            if self._robot_outside_of_boundary_box(rs_state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3]):
                reward = -30.0
                done = True
                info["final_status"] = "out of boundary"
                print("Robot out of boundary")

        # Target Reached
        if euclidean_dist_2d < self.distance_threshold:
            reward = 20
            done = True
            info["final_status"] = "success"
            info["elapsed_time"] = rs_state[RS_ROSTIME] - self.episode_start_time
            print("Target Reached!")
            print("Episode Length: ", str(self.elapsed_steps))

        # Time step exceeded
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info["final_status"] = "max_steps_exceeded"
            print("max_step_exceeded")
            print("Distance from target: ", str(euclidean_dist_2d))

        return reward, done, info

    def render(self):
        pass

    def _get_robot_server_state_len(self):
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """

        target = [0.0] * 3
        base_pose = [0.0] * 3
        base_twist = [0.0] * 2
        scan = [0.0] * self.laser_len
        collision = False
        rostime = [0.0]
        obstacles = [0.0 for i in range(0, 3 * NUM_OBSTACLES)]

        rs_state = target + base_pose + base_twist + scan + [collision] + rostime + obstacles

        return len(rs_state)

    def _sim_robot_collision(self, rs_state):
        """Get status of simulated collision sensor.

        Used only for simulated Robot Server.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if the robot is in collision.

        """

        if rs_state[RS_COLLISION] == 1:
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
        safety_radius = 0.5

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
        if np.sqrt((obst_pose[0] - rs_state[RS_TARGET]) ** 2 + (obst_pose[1] - rs_state[RS_TARGET + 1]) ** 2) < 0.8:
            return False
        for i in range(0, len(self.sim_obstacles)):
            if np.sqrt((self.sim_obstacles[i][0] - obst_pose[0]) ** 2 + (self.sim_obstacles[i][1] - obst_pose[1]) ** 2) < DIST_BTW_OBSTACLES:
                return False
        return True


class Obstacle_Avoidance_Jackal_Kinova_Sim(Obstacle_Avoidance_Jackal_Kinova, Simulation):
    cmd = "roslaunch jackal_kinova_robot_server sim_robot_server.launch world_name:=lab_6by9_obst_3.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=True, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Obstacle_Avoidance_Jackal_Kinova.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class Obstacle_Avoidance_Jackal_Kinova_Rob(Obstacle_Avoidance_Jackal_Kinova):
    real_robot = True


class Fixed_Obstacle_Avoidance_Jackal_Kinova(No_Obstacle_Avoidance_Jackal_Kinova):
    def step(self, action):
        action = action.astype(np.float32)

        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        # Convert environment action to Robot Server action
        rs_action = copy.deepcopy(action)

        # Scale action (use when the normalized action space is used)
        # rs_action = np.multiply(action, self.max_vel)

        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state
        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # Assign reward
        reward, done, info = self._reward(rs_state=rs_state, action=action)

        return self.state, reward, done, info

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
        self.acc_penalty = 0.0
        self.zero_vel_penalty = 0.0

        self.prev_lin_vel, self.prev_ang_vel = 0.0, 0.0
        self.episode_start_time = 0.0
        self.prev_rostime = 0.0

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        env_num = np.random.randint(0, len(start_points))
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

        self.max_episode_steps = e_lengths[env_num]

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
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        return self.state

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}

        # Reward base: Distance to target

        # Calculate distance to the target
        target_coords = np.array([rs_state[RS_TARGET], rs_state[RS_TARGET + 1]])
        robot_coords = np.array([rs_state[RS_ROBOT_POSE], rs_state[RS_ROBOT_POSE + 1]])
        euclidean_dist_2d = np.linalg.norm(target_coords - robot_coords, axis=-1)
        # print("Target coordinate: {:.3f}, {:.3f}".format(rs_state[RS_TARGET], rs_state[RS_TARGET + 1]))
        # print("Distance: {:.3f}".format(euclidean_dist_2d))
        # print("Robot heading: {:.3f}".format(rs_state[RS_ROBOT_POSE + 2]))
        # print("Heading Error:: {:.3f}".format(self.state[1]))

        base_reward = -30 * euclidean_dist_2d

        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        if abs(self.state[1]) < np.pi / 3:
            reward += 1.3
        else:
            reward -= 1

        # Negative rewards
        # if rs_state[RS_ROBOT_TWIST] < 0.01 and self.zero_vel_penalty > -200:
        #     self.zero_vel_penalty = self.zero_vel_penalty - 5
        #     reward = reward - 5

        # Spinning in plance
        if action[0] < 0.01 and abs(action[1]) > 0.0:
            reward -= 0.3

        # Power used by the motors
        # linear_power = abs(action[0] * 0.15)
        # angular_power = abs(action[1] * 0.3)
        # reward -= linear_power
        # reward -= angular_power

        if self.prev_rostime == 0.0:
            self.prev_rostime = rs_state[RS_ROSTIME]
            self.episode_start_time = rs_state[RS_ROSTIME]
        else:
            if self.acc_penalty > -200:
                # High acceleration
                # 1: Continous Penalty
                # r = 10 * (abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) + abs(rs_state[RS_ROBOT_TWIST] - self.prev_ang_vel))

                # 2: Discrete Penalty
                timediff = rs_state[RS_ROSTIME] - self.prev_rostime
                # r = 5
                # if abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) / timediff > self.apf_util.get_max_lin_acc():
                #     self.acc_penalty = self.acc_penalty - r
                # if abs(rs_state[RS_ROBOT_TWIST + 1] - self.prev_ang_vel) / timediff > self.apf_util.get_max_ang_acc():
                #     self.acc_penalty = self.acc_penalty - r

                # reward = reward - r
            self.prev_rostime = rs_state[RS_ROSTIME]

        self.prev_lin_vel = rs_state[RS_ROBOT_TWIST]
        self.prev_ang_vel = rs_state[RS_ROBOT_TWIST + 1]

        # Long path length (episode length)
        reward -= 0.7

        if not self.real_robot:
            # End episode if robot is collides with an object.
            if self._sim_robot_collision(rs_state):
                reward = -30.0
                done = True
                info["final_status"] = "collision"
                print("collision occured")
                print("Episode Length: ", str(self.elapsed_steps))

            if self._min_laser_reading_below_threshold(rs_state):
                reward = -10.0
                done = True
                info["final_status"] = "front collision"
                print("front blocked")
                print("Episode Length: ", str(self.elapsed_steps))

        # Target Reached
        if euclidean_dist_2d < self.distance_threshold:
            reward = 20
            done = True
            info["final_status"] = "success"
            info["elapsed_time"] = rs_state[RS_ROSTIME] - self.episode_start_time
            print("Target Reached!")
            print("Episode Length: ", str(self.elapsed_steps))

        # Time step exceeded
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info["final_status"] = "max_steps_exceeded"
            print("max_step_exceeded")
            print("Distance from target: ", str(euclidean_dist_2d))

        return reward, done, info

    def render(self):
        pass

    def _get_robot_server_state_len(self):
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """

        target = [0.0] * 3
        base_pose = [0.0] * 3
        base_twist = [0.0] * 2
        scan = [0.0] * self.laser_len
        collision = False
        rostime = [0.0]

        rs_state = target + base_pose + base_twist + scan + [collision] + rostime

        return len(rs_state)

    def _sim_robot_collision(self, rs_state):
        """Get status of simulated collision sensor.

        Used only for simulated Robot Server.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if the robot is in collision.

        """

        if rs_state[RS_COLLISION] == 1:
            return True
        else:
            return False

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


class Fixed_Obstacle_Avoidance_Jackal_Kinova_Sim(Fixed_Obstacle_Avoidance_Jackal_Kinova, Simulation):
    cmd = "roslaunch jackal_kinova_robot_server sim_robot_server.launch world_name:=learning_world_4.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=True, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Fixed_Obstacle_Avoidance_Jackal_Kinova.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class Fixed_Obstacle_Avoidance_Jackal_Kinova_Rob(Fixed_Obstacle_Avoidance_Jackal_Kinova):
    real_robot = True
