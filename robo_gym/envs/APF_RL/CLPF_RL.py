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
DIST_BTW_OBSTACLES = 2.5
# State Indecies

RS_TARGET = 0
RS_WEIGHTS = RS_TARGET + 3
RS_SCAN = RS_WEIGHTS + 2
RS_ROBOT_POSE = RS_SCAN + 811  # Laser scan length of jackal
RS_ROBOT_TWIST = RS_ROBOT_POSE + 3
RS_FORCES = RS_ROBOT_TWIST + 2
RS_COLLISION = RS_FORCES + 9
RS_ROSTIME = RS_COLLISION + 1
RS_DETECTED_OBS = RS_ROSTIME + 1
RS_DIST_MOVED = RS_DETECTED_OBS + 1
RS_ACC_LIN = RS_DIST_MOVED + 1
RS_ACC_ANG = RS_ACC_LIN + 1
RS_PDGAINS = RS_DETECTED_OBS + 1

start_points = [
    [1.5, 10.0, 0.0],
    [1.5, 4.0, 0.0],
    [1.5, -4.0, 0.0],
    [1.5, -10.0, 0.0],
    [-9.5, 10.0, 0.0],
    [-9.5, 4.0, 0],
    [-9.5, -4.0, 0.0],
    [-9.5, -10.0, 0.0],
]

target_points = [
    [8.75, 10.0, 0.0],
    [8.75, 4.0, 0.0],
    [8.75, -4.0, 0.0],
    [8.75, -10.0, 0.0],
    [-2.75, 10.0, 0.0],
    [-2.75, 3.0, 0.0],
    [-2.75, -4.0, 0.0],
    [-2.75, -10.0, 0.0],
]


class Clustered_APF_Jackal_Kinova(gym.Env):
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
    laser_downsample_len = 128
    max_episode_steps = 600
    learning = True

    def __init__(self, rs_address=None, **kwargs):
        self.jackal_kinova = jackal_kinova_utils.Jackal_Kinova()
        self.apf_util = apf_env_utils.APF()

        # KP, ETA, obstacle_ratio
        self.action_low = [0.01, 0.01, 0.001]
        self.action_high = [1, 1, 1]

        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()

        self.initialize()

        self.action_space = spaces.MultiDiscrete([4, 5, 7, 100])
        self.seed()
        self.distance_threshold = 0.3

        # Maximum linear velocity (m/s) of Robot
        max_lin_vel = self.jackal_kinova.get_max_lin_vel()
        # Maximum angular velocity (rad/s) of Robot
        max_ang_vel = self.jackal_kinova.get_max_ang_vel()
        self.max_vel = np.array([max_lin_vel, max_ang_vel])

        self.episode_start_time = 0.0
        self.prev_lin_vel = 0.0
        self.prev_ang_vel = 0.0
        self.prev_rostime = 0.0

        self.acc_penalty = 0.0
        self.zero_vel_penalty = 0.0

        self.distance_moved = 0.0
        self.total_acc_lin = 0.0
        self.total_acc_ang = 0.0

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
        self.acc_penalty = 0.0
        self.zero_vel_penalty = 0.0

        self.prev_lin_vel = 0.0
        self.prev_ang_vel = 0.0
        self.episode_start_time = 0.0
        self.prev_rostime = 0.0
        self.distance_moved = 0.0
        self.acc_lin = 0.0
        self.acc_ang = 0.0

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # env_num = np.random.randint(0, 4)
        env_num = 7
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
        reward = 0.0
        done = False
        info = {}
        r = 0.0

        self.distance_moved += rs_state[RS_DIST_MOVED]
        self.total_acc_lin += rs_state[RS_ACC_LIN]
        self.total_acc_ang += rs_state[RS_ACC_ANG]
        # Reward base: Distance to target

        # Calculate distance to the target
        target_coords = np.array([rs_state[RS_TARGET], rs_state[RS_TARGET + 1]])
        robot_coords = np.array([rs_state[RS_ROBOT_POSE], rs_state[RS_ROBOT_POSE + 1]])
        euclidean_dist_2d = np.linalg.norm(target_coords - robot_coords, axis=-1)

        # Calculate Attractive, Repulsive force direction
        # polar_r, att_theta = utils.cartesian_to_polar_2d(
        #     x_target=0,
        #     y_target=0,
        #     x_origin=rs_state[RS_FORCES],
        #     y_origin=rs_state[RS_FORCES + 1],
        # )

        # polar_r, rep_theta = utils.cartesian_to_polar_2d(
        #     x_target=0,
        #     y_target=0,
        #     x_origin=rs_state[RS_FORCES + 3],
        #     y_origin=rs_state[RS_FORCES + 4],
        # )

        base_reward = -60.0 * euclidean_dist_2d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        # Negative rewards

        # Attractive force offset
        # if abs(att_theta - rep_theta) < np.pi / 6:
        #     if action[2] == 0.0:
        #         reward -= 2.0
        # else:
        #     if action[2] != 0.0:
        #         reward -= 1.0

        if self.prev_rostime == 0.0:
            self.prev_rostime = rs_state[RS_ROSTIME]
            self.episode_start_time = rs_state[RS_ROSTIME]
            self.prev_base_pose = rs_state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3]
        else:
            # High acceleration
            timediff = rs_state[RS_ROSTIME] - self.prev_rostime
            self.acc_lin = abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) / timediff
            self.acc_ang = abs(rs_state[RS_ROBOT_TWIST + 1] - self.prev_ang_vel) / timediff

            # 1: Continous Penalty
            # r = 10 * (abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) + abs(rs_state[RS_ROBOT_TWIST] - self.prev_ang_vel))
            r = (abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) / timediff) + (
                abs(rs_state[RS_ROBOT_TWIST + 1] - self.prev_ang_vel) / timediff
            )

            # 2: Discrete Penalty
            # if (
            #     abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) / timediff
            #     > self.apf_util.get_max_lin_acc()
            # ):
            #     r = 0.5
            # if (
            #     abs(rs_state[RS_ROBOT_TWIST + 1] - self.prev_ang_vel) / timediff
            #     > self.apf_util.get_max_ang_acc()
            # ):
            #     r = 0.5

            if r > 0.5:
                r = 0.5
            reward -= r

        if rs_state[RS_DETECTED_OBS] > 5:
            if action[2] > 150:
                reward -= 0.5

        if rs_state[RS_DETECTED_OBS] < 5:
            if action[2] < 200:
                reward -= 0.5

        self.prev_rostime = rs_state[RS_ROSTIME]
        self.prev_lin_vel = rs_state[RS_ROBOT_TWIST]
        self.prev_ang_vel = rs_state[RS_ROBOT_TWIST + 1]

        # path length (episode length)
        # reward -= 60.0 / self.max_episode_steps

        if not self.real_robot:
            # End episode if robot is collides with an object.
            if self._sim_robot_collision(rs_state):
                reward = -80.0
                done = True
                info["final_status"] = "collision"
                self.total_acc_lin = 0.0
                self.total_acc_ang = 0.0
                self.distance_moved = 0.0
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
                self.total_acc_lin = 0.0
                self.total_acc_ang = 0.0
                self.distance_moved = 0.0

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
            info["acc_lin"] = self.total_acc_lin
            info["acc_ang"] = self.total_acc_ang
            info["distance_moved"] = self.distance_moved

            self.total_acc_lin = 0.0
            self.total_acc_ang = 0.0
            self.distance_moved = 0.0

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
            self.total_acc_lin = 0.0
            self.total_acc_ang = 0.0
            self.distance_moved = 0.0

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
        reward, done, info = self._reward(rs_state=rs_state, action=action)

        return self.state, reward, done, info

    def render(self):
        pass

    def _denormalize_actions(self, action):
        rs_action = np.zeros(4, dtype=np.float32)
        att = [60, 120, 180, 240]
        rep = [50, 100, 150, 200, 250]
        rs_action[0] = att[action[0]]
        rs_action[1] = rep[action[1]]
        rs_action[2] = -np.pi + (action[2] * np.pi / 3)
        rs_action[3] = (action[3] + 1) / 100.0
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
        detected_obs = [0.0]
        distance_moved = [0.0]
        acc_lin = [0.0]
        acc_ang = [0.0]

        rs_state = (
            target
            + apf_weights
            + scan
            + robot_pose
            + robot_twist
            + forces
            + collision
            + rostime
            + detected_obs
            + distance_moved
            + acc_lin
            + acc_ang
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
        detected_obs = [0.0]
        env_state = target_polar_coordinates + scan + weights + detected_obs + forces + robot_twist

        return len(env_state)

    # def _get_start_pose(self):
    #     """Get initial robot coordinates.

    #     For the real robot the initial coordinates are its current coordinates
    #     whereas for the simulated robot the initial coordinates are
    #     randomly generated.

    #     Returns:
    #         numpy.array: [x,y,yaw] robot initial coordinates.
    #     """

    #     if self.real_robot:
    #         # Take current robot position as start position
    #         start_pose = self.client.get_state_msg().state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3]
    #     else:
    #         # Create random starting position
    #         x = self.np_random.uniform(low=-2.0, high=2.0)
    #         y = self.np_random.uniform(low=-2.0, high=2.0)
    #         yaw = self.np_random.uniform(low=-np.pi / 2, high=np.pi / 2)

    #         # Using x=0, y=0 for testing purpose
    #         x, y = 0, 0

    #         start_pose = [x, y, yaw]

    #     return start_pose

    # def _get_target(self, robot_coordinates):
    #     """Generate coordinates of the target at a minimum distance from the robot.

    #     Args:
    #         robot_coordinates (list): [x,y,yaw] coordinates of the robot.

    #     Returns:
    #         numpy.array: [x,y,yaw] coordinates of the target.

    #     """

    #     x_t = self.np_random.uniform(low=7.5, high=8.2)
    #     y_t = self.np_random.uniform(low=-2.5, high=2.5)
    #     yaw_t = 0.0
    #     # target_dist = np.linalg.norm(np.array([x_t, y_t]) - np.array(robot_coordinates[0:2]), axis=-1)

    #     return [x_t, y_t, yaw_t]

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
                np.array([rs_state[RS_DETECTED_OBS]]),
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

        # LaserScan
        max_laser = np.full(self.laser_downsample_len, np.inf)  # Using inf due to sensor noise
        min_laser = np.full(self.laser_downsample_len, 0.0)

        # Weights
        max_weights = np.array([np.inf, np.inf])
        min_weights = np.array([0.0, 0.0])

        # Number of Detected Obstacle
        max_detected_obst = np.array([np.inf])
        min_detected_obst = np.array([0.0])

        # f_x, f_y, |f|
        max_forces = np.array([np.inf for i in range(0, 9)])
        min_forces = np.array([-np.inf for i in range(0, 9)])

        # Robot velocity range used to determine if there is an error in the sensor readings
        max_lin_vel = self.jackal_kinova.get_max_lin_vel() + vel_tolerance
        min_lin_vel = self.jackal_kinova.get_min_lin_vel() - vel_tolerance
        max_ang_vel = self.jackal_kinova.get_max_ang_vel() + vel_tolerance
        min_ang_vel = self.jackal_kinova.get_min_ang_vel() - vel_tolerance

        max_vel = np.array([max_lin_vel, max_ang_vel])
        min_vel = np.array([min_lin_vel, min_ang_vel])

        # Definition of environment observation_space
        obs_space_max = np.concatenate(
            (
                max_target_coords,
                max_laser,
                max_weights,
                max_detected_obst,
                max_forces,
                max_vel,
            )
        )
        obs_space_min = np.concatenate(
            (
                min_target_coords,
                min_laser,
                min_weights,
                min_detected_obst,
                min_forces,
                min_vel,
            )
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
        width = 20
        height = 14

        if (
            robot_coordinates[0] < -2
            or robot_coordinates[0] > width
            or np.absolute(robot_coordinates[1]) > (height / 2)
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
        threshold = 0.2
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
        safety_radius = 0.9

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

    def _generate_obstacles_positions(self):
        """Generate random positions for 3 obstacles.

        Used only for simulated Robot Server.

        """

        self.sim_obstacles = []
        for i in range(0, NUM_OBSTACLES):
            if i == 0:
                pose = [self.np_random.uniform(low=2.5, high=4), 0, 0]
            else:
                pose = self.generate_pos()
            while not self.is_valid_obstacle(pose):
                pose = self.generate_pos()
            self.sim_obstacles.append(pose)

    def generate_pos(self):
        x = self.np_random.uniform(low=1.2, high=6.7)
        y = self.np_random.uniform(low=-2.1, high=2.1)
        yaw = self.np_random.uniform(low=-np.pi, high=np.pi)
        return [x, y, yaw]

    def is_valid_obstacle(self, obst_pose):
        if len(self.sim_obstacles) == 0:
            return True
        else:
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


class Clustered_APF_Jackal_Kinova_Sim(Clustered_APF_Jackal_Kinova, Simulation):
    cmd = "roslaunch apf_robot_server sim_clpf_robot_server.launch world_name:=final_rooms.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=True, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Clustered_APF_Jackal_Kinova.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class Clustered_APF_Jackal_Kinova_Sim_Experiment(Clustered_APF_Jackal_Kinova_Sim):
    learning = False


class Clustered_APF_Jackal_Kinova_Rob(Clustered_APF_Jackal_Kinova):
    real_robot = True


class Clustered_APF_with_PD_Jackal_Kinova(Clustered_APF_Jackal_Kinova):
    def initialize(self):
        # linear_kd, angular_kp, angular_kd
        self.action_low.extend([0, 0, 0])
        self.action_high.extend([0.9, 1, 1])

        return

    def _denormalize_actions(self, action):
        rs_action = np.zeros(3, dtype=np.float32)
        rs_action[0] = 100 * action[0]
        rs_action[1] = 500 * action[1]
        rs_action[2] = action[2]
        rs_action[3] = action[3]
        rs_action[4] = 10 * action[4]
        rs_action[5] = 10 * action[5]
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
        obstacles = [0.0 for i in range(3 * NUM_OBSTACLES)]
        rostime = [0.0]
        detected_obs = [0]
        pd_gains = [0.0 for i in range(0, 3)]

        rs_state = (
            target
            + apf_weights
            + scan
            + robot_pose
            + robot_twist
            + forces
            + [collision]
            + obstacles
            + rostime
            + detected_obs
            + pd_gains
        )

        return len(rs_state)


class Clustered_APF_with_PD_Jackal_Kinova_Sim(Clustered_APF_with_PD_Jackal_Kinova, Simulation):
    cmd = "roslaunch apf_robot_server sim_clpf_robot_server_with_pd.launch world_name:=obst_4_6by9.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=True, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Clustered_APF_with_PD_Jackal_Kinova.__init__(
            self, rs_address=self.robot_server_ip, **kwargs
        )


class Clustered_APF_with_PD_Jackal_Kinova_Rob(Clustered_APF_with_PD_Jackal_Kinova):
    real_robot = True
