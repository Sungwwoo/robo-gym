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
RS_BASE_POSE = 3
RS_BASE_TWIST = 6
RS_SCAN = 8
RS_COLLISION = 8 + 811  # Raser Scan Length
RS_OBSTACLES = RS_COLLISION + 1


class Jackal_Kinova_Env(gym.Env):
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
    max_episode_steps = 9000

    def __init__(self, rs_address=None, **kwargs):
        self.jackal_kinova = jackal_kinova_utils.Jackal_Kinova()

        action_low = [self.jackal_kinova.get_min_lin_vel(), self.jackal_kinova.get_min_ang_vel()]  # linear, angular
        action_high = [self.jackal_kinova.get_max_lin_vel(), self.jackal_kinova.get_max_ang_vel()]  # linear, angular

        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()
        self.action_space = spaces.Box(low=np.array(action_low), high=np.array(action_high), dtype=np.float32)
        self.seed()
        self.distance_threshold = 0.3
        self.min_target_dist = 1.0
        # Maximum linear velocity (m/s) of Husky
        max_lin_vel = self.jackal_kinova.get_max_lin_vel()
        # Maximum angular velocity (rad/s) of Husky
        max_ang_vel = self.jackal_kinova.get_max_ang_vel()
        self.max_vel = np.array([max_lin_vel, max_ang_vel])
        self.prev_lin_vel = 0
        self.prev_ang_vel = 0

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

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set Robot starting position
        if start_pose:
            assert len(start_pose) == 3
        else:
            start_pose = self._get_start_pose()

        rs_state[3:6] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose) == 3
        else:
            target_pose = self._get_target(start_pose)
        rs_state[0:3] = target_pose

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
        return 0, False, {}

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
        husky_pose = [0.0] * 3
        husky_twist = [0.0] * 2
        scan = [0.0] * self.laser_len
        collision = False
        obstacles = [0.0] * 21
        rs_state = target + husky_pose + husky_twist + scan + [collision] + obstacles  # 3+3+2+716+1+9 = 560

        return len(rs_state)

    def _get_env_state_len(self):
        """Get length of the environment state.

        Describes the composition of the environment state and returns
        its length.

        Returns:
            int: Length of the environment state

        """

        target_polar_coordinates = [0.0] * 2
        husky_twist = [0.0] * 2
        laser = [0.0] * self.laser_len
        env_state = target_polar_coordinates + husky_twist + laser

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
            x = self.np_random.uniform(low=-2.0, high=2.0)
            y = self.np_random.uniform(low=-2.0, high=2.0)
            yaw = self.np_random.uniform(low=-np.pi, high=np.pi)
            start_pose = [x, y, yaw]

        return start_pose

    def _get_target(self, robot_coordinates):
        """Generate coordinates of the target at a minimum distance from the robot.

        Args:
            robot_coordinates (list): [x,y,yaw] coordinates of the robot.

        Returns:
            numpy.array: [x,y,yaw] coordinates of the target.

        """

        target_far_enough = False
        while not target_far_enough:
            x_t = self.np_random.uniform(low=-1.0, high=1.0)
            y_t = self.np_random.uniform(low=-1.0, high=1.0)
            yaw_t = 0.0
            target_dist = np.linalg.norm(np.array([x_t, y_t]) - np.array(robot_coordinates[0:2]), axis=-1)

            if target_dist >= self.min_target_dist:
                target_far_enough = True

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
            x_target=rs_state[0],
            y_target=rs_state[1],
            x_origin=rs_state[3],
            y_origin=rs_state[4],
        )
        # Rotate origin of polar coordinates frame to be matching with robot frame and normalize to +/- pi
        polar_theta = utils.normalize_angle_rad(polar_theta - rs_state[5])

        # Get Laser scanners data
        raw_laser_scan = rs_state[8 : 8 + self.laser_len]

        # Downsampling of laser values by picking every n-th value
        if self.laser_len > 0:
            laser = utils.downsample_list_to_len(raw_laser_scan, self.laser_len)
            # Compose environment state
            state = np.concatenate((np.array([polar_r, polar_theta]), rs_state[6:8], laser))
        else:
            # Compose environment state
            state = np.concatenate((np.array([polar_r, polar_theta]), rs_state[6:8]))

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
        vel_tolerance = 0.5
        # Robot velocity range used to determine if there is an error in the sensor readings
        max_lin_vel = self.jackal_kinova.get_max_lin_vel() + vel_tolerance
        min_lin_vel = self.jackal_kinova.get_min_lin_vel() - vel_tolerance
        max_ang_vel = self.jackal_kinova.get_max_ang_vel() + vel_tolerance
        min_ang_vel = self.jackal_kinova.get_min_ang_vel() - vel_tolerance
        max_vel = np.array([max_lin_vel, max_ang_vel])
        min_vel = np.array([min_lin_vel, min_ang_vel])

        # Laser readings range
        max_laser = np.full(self.laser_len, 25.0)
        min_laser = np.full(self.laser_len, 0.0)

        # Definition of environment observation_space
        max_obs = np.concatenate((max_target_coords, max_vel, max_laser))
        min_obs = np.concatenate((min_target_coords, min_vel, min_laser))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

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
        height = 20

        if np.absolute(robot_coordinates[0]) > (width / 2) or np.absolute(robot_coordinates[1] > (height / 2)):
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

        if rs_state[8 + self.laser_len] == 1:
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

        threshold = 0.05
        if min(rs_state[8 : 8 + self.laser_len]) < threshold:
            return True
        else:
            return False


class Obstacle_7_Avoidance_Jackal_Kinova(Jackal_Kinova_Env):
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

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set Robot starting position
        if start_pose:
            assert len(start_pose) == 3
        else:
            start_pose = self._get_start_pose()

        rs_state[3:6] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose) == 3
        else:
            target_pose = self._get_target(start_pose)
        rs_state[0:3] = target_pose

        # Generate obstacles positions
        self._generate_obstacles_positions()
        start_idx = 8 + self.laser_len + 1
        for i in range(0, 7):
            rs_state[start_idx + 3 * i : start_idx + 3 * (i + 1)] = self.sim_obstacles[i]

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
        linear_power = 0
        angular_power = 0

        # Calculate distance to the target
        target_coords = np.array([rs_state[0], rs_state[1]])
        husky_coords = np.array([rs_state[3], rs_state[4]])
        euclidean_dist_2d = np.linalg.norm(target_coords - husky_coords, axis=-1)

        # Reward base
        base_reward = -100 * euclidean_dist_2d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        # Negative rewards

        # High acceleration
        # 1: Continous Penalty
        # reward = -10 * (abs(rs_state[RS_BASE_TWIST] - self.prev_lin_vel) + abs(rs_state[RS_ROBOT_TWIST] - self.prev_ang_vel))

        # 2: Discrete Penalty
        if abs(rs_state[RS_BASE_TWIST] - self.prev_lin_vel) > self.jackal_kinova.get_max_lin_acc():
            reward = reward - 20
        if abs(rs_state[RS_BASE_TWIST + 1] - self.prev_ang_vel) > self.jackal_kinova.get_max_ang_acc():
            reward = reward - 40

        self.prev_lin_vel = rs_state[RS_BASE_TWIST]
        self.prev_ang_vel = rs_state[RS_BASE_TWIST + 1]

        # Long path length (episode length)
        reward = reward - 0.001 * self.elapsed_steps

        # TODO: Distance to obstacle

        # End episode if robot is collides with an object, if it is too close
        # to an object.
        if not self.real_robot:
            if self._sim_robot_collision(rs_state):
                reward = -200.0
                done = True
                info["final_status"] = "collision"
                print("contact_collision")
                print("Distance from target: ", str(euclidean_dist_2d))

            if self._min_laser_reading_below_threshold(rs_state):
                reward = -200
                done = True
                info["final_status"] = "collision"
                print("laser_collision")
                print("Distance from target: ", str(euclidean_dist_2d))

            if self._robot_close_to_sim_obstacle(rs_state):
                reward = -200.0
                done = True
                info["final_status"] = "collision"
                print("obs_collision")
                print("Distance from target: ", str(euclidean_dist_2d))

            if self._robot_outside_of_boundary_box(rs_state[RS_BASE_POSE : RS_BASE_POSE + 3]):
                reward = -200.0
                done = True
                info["final_status"] = "out of boundary"
                print("Robot out of boundary")

        if euclidean_dist_2d < self.distance_threshold:
            reward = 500
            done = True
            info["final_status"] = "success"
            print("Target Reached!")

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info["final_status"] = "max_steps_exceeded"
            print("max_step_exceeded")
            print("Distance from target: ", str(euclidean_dist_2d))

        return reward, done, info

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
            yaw = self.np_random.uniform(low=-np.pi, high=np.pi)

            x, y, yaw = 0, 0, 0
            start_pose = [x, y, yaw]

        return start_pose

    def _get_target(self, robot_coordinates):
        """Generate coordinates of the target at a minimum distance from the robot.

        Args:
            robot_coordinates (list): [x,y,yaw] coordinates of the robot.

        Returns:
            numpy.array: [x,y,yaw] coordinates of the target.

        """

        x_t = self.np_random.uniform(low=15.0, high=17.0)
        y_t = self.np_random.uniform(low=-3.0, high=3.0)
        yaw_t = 0.0

        return [x_t, y_t, yaw_t]

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
        robot_corners = self.jackal_kinova.get_corners_positions(rs_state[3], rs_state[4], rs_state[5])

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

    # TODO: Parmeterize the number of obstacles
    def _generate_obstacles_positions(
        self,
    ):
        """Generate random positions for 3 obstacles.

        Used only for simulated Robot Server.

        """

        self.sim_obstacles = []
        for i in range(0, 7):
            pose = self.generate_pos()
            while not self.is_valid_obstacle(pose):
                pose = self.generate_pos()
            self.sim_obstacles.append(pose)

    def generate_pos(self):
        x = self.np_random.uniform(low=2.0, high=14.0)
        y = self.np_random.uniform(low=-6.0, high=6.0)
        yaw = self.np_random.uniform(low=-np.pi, high=np.pi)
        return [x, y, yaw]

    def is_valid_obstacle(self, obst_pose):
        if len(self.sim_obstacles) == 0:
            return True
        else:
            for i in range(0, len(self.sim_obstacles)):
                if np.sqrt((self.sim_obstacles[i][0] - obst_pose[0]) ** 2 + (self.sim_obstacles[i][1] - obst_pose[1]) ** 2) < 2.0:
                    return False
            return True

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

        if robot_coordinates[0] < -2 or robot_coordinates[0] > width or np.absolute(robot_coordinates[1]) > (height / 2):
            return True
        else:
            return False


class Obstacle_7_Avoidance_Jackal_Kinova_Sim(Obstacle_7_Avoidance_Jackal_Kinova, Simulation):
    cmd = "roslaunch jackal_kinova_robot_server sim_robot_server.launch world_name:=pf_test_world_4.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=True, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Obstacle_7_Avoidance_Jackal_Kinova.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class Obstacle_7_Avoidance_Jackal_Kinova_Rob(Obstacle_7_Avoidance_Jackal_Kinova):
    real_robot = True
