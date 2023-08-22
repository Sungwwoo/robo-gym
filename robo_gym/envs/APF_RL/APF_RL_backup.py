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
RS_WEIGHTS = 3
RS_ROBOT_POSE = 5
RS_ROBOT_TWIST = 8
RS_FORCES = 10
RS_COLLISION = 13
RS_OBSTACLES = 14


# env_state
# target_polar_coordinates = [0.0] * 2
# robot_twist = [0.0] * 2
ENV_TARGET = 0
ENV_ROBOT_TWIST = 2


class APF_Env(gym.Env):
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
        self.apf_util = apf_env_utils.APF()

        # KP, ETA
        action_low = [5, 100]
        action_high = [100, 500]

        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()
        self.action_space = spaces.Box(low=np.array(action_low), high=np.array(action_high), dtype=np.float32)

        self.seed()
        self.distance_threshold = 0.3

        # Maximum linear velocity (m/s) of Robot
        max_lin_vel = self.jackal_kinova.get_max_lin_vel()
        # Maximum angular velocity (rad/s) of Robot
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
        """Environment reset using set_state() of RosBridge.

        Args:
            start_pose (list[3] or np.array[3]): [x,y,yaw] initial robot position.
            target_pose (list[3] or np.array[3]): [x,y,yaw] target robot position.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.prev_base_reward = None
        self.prev_lin_vel = 0
        self.prev_ang_vel = 0

        # Initialize environment state

        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set target position
        if target_pose:
            assert len(target_pose) == 3
        else:
            target_pose = self._get_target(start_pose)

        rs_state[RS_TARGET : RS_TARGET + 3] = target_pose

        # Set initial weights
        rs_state[RS_WEIGHTS : RS_WEIGHTS + 2] = [20, 10]

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state=rs_state.tolist())

        # Set states to robot_server.RosBridge.set_state()
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
        """Send action, get rs_state, and calculate reward"""
        action = action.astype(np.float32)

        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        # Convert environment action to Robot Server action
        rs_action = copy.deepcopy(action)

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

    def _get_robot_server_state_len(self):
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """

        target = [0.0] * 3
        apf_weights = [0.0] * 2
        robot_pose = [0.0] * 3
        robot_twist = [0.0] * 2
        forces = [0.0] * 3
        collision = False
        obstacles = [0.0] * 21
        rs_state = target + apf_weights + robot_pose + robot_twist + forces + [collision] + obstacles

        return len(rs_state)

    def _get_env_state_len(self):
        """Get length of the environment state.

        Describes the composition of the environment state and returns
        its length.

        Returns:
            int: Length of the environment state

        """

        target_polar_coordinates = [0.0] * 2
        robot_twist = [0.0] * 2  # Linear, Angular
        env_state = target_polar_coordinates + robot_twist

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
            start_pose = self.client.get_state_msg().state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3]
        else:
            # Create random starting position
            x = self.np_random.uniform(low=-2.0, high=2.0)
            y = self.np_random.uniform(low=-2.0, high=2.0)
            yaw = self.np_random.uniform(low=-np.pi, high=np.pi)

            # Using x=0, y=0 for testing purpose
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

        x_t = self.np_random.uniform(low=15.0, high=17.0)
        y_t = self.np_random.uniform(low=-3.0, high=3.0)
        yaw_t = 0.0
        # target_dist = np.linalg.norm(np.array([x_t, y_t]) - np.array(robot_coordinates[0:2]), axis=-1)

        return [x_t, y_t, yaw_t]

    # TODO: edit states
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

        state = np.concatenate((np.array([polar_r, polar_theta]), rs_state[RS_ROBOT_TWIST : RS_ROBOT_TWIST + 2]))

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

        # Definition of environment observation_space
        obs_space_max = np.concatenate((max_target_coords, max_vel))
        obs_space_min = np.concatenate((min_target_coords, min_vel))

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

        if robot_coordinates[0] < -2 or robot_coordinates[0] > width or np.absolute(robot_coordinates[1]) > (height / 2):
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

        return False


class Basic_APF_Jackal_Kinova(APF_Env):
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

        self.prev_lin_vel = 0
        self.prev_ang_vel = 0

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
        self._generate_obstacles_positions()
        for i in range(0, 7):
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

        # Reward base: Distance to target

        # Calculate distance to the target
        target_coords = np.array([rs_state[RS_TARGET], rs_state[RS_TARGET + 1]])
        robot_coords = np.array([rs_state[RS_ROBOT_POSE], rs_state[RS_ROBOT_POSE + 1]])
        euclidean_dist_2d = np.linalg.norm(target_coords - robot_coords, axis=-1)

        base_reward = -100 * euclidean_dist_2d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        # Negative rewards

        # High acceleration

        # 1: Continous Penalty
        # reward = -10 * (abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) + abs(rs_state[RS_ROBOT_TWIST] - self.prev_ang_vel))

        # 2: Discrete Penalty
        if abs(rs_state[RS_ROBOT_TWIST] - self.prev_lin_vel) > self.apf_util.get_max_lin_acc():
            reward = reward - 20
        if abs(rs_state[RS_ROBOT_TWIST + 1] - self.prev_ang_vel) > self.apf_util.get_max_ang_acc():
            reward = reward - 40

        self.prev_lin_vel = rs_state[RS_ROBOT_TWIST]
        self.prev_ang_vel = rs_state[RS_ROBOT_TWIST + 1]

        # Long path length (episode length?)
        reward = reward - 0.005 * self.elapsed_steps
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

            if self._robot_close_to_sim_obstacle(rs_state):
                reward = -200.0
                done = True
                info["final_status"] = "collision"
                print("obs_collision")
                print("Distance from target: ", str(euclidean_dist_2d))

            if self._robot_outside_of_boundary_box(rs_state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3]):
                reward = -200.0
                done = True
                info["final_status"] = "out of boundary"
                print("Robot out of boundary")

        # Target Reached
        if euclidean_dist_2d < self.distance_threshold:
            reward = 500
            done = True
            info["final_status"] = "success"
            print("Target Reached!")

        # Time step exceeded
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info["final_status"] = "max_steps_exceeded"
            print("max_step_exceeded")
            print("Distance from target: ", str(euclidean_dist_2d))

        return reward, done, info

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
        robot_corners = self.jackal_kinova.get_corners_positions(rs_state[RS_ROBOT_POSE], rs_state[RS_ROBOT_POSE + 1], rs_state[RS_ROBOT_POSE + 2])

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


class Basic_APF_Jackal_Kinova_Sim(Basic_APF_Jackal_Kinova, Simulation):
    cmd = "roslaunch apf_robot_server sim_robot_server.launch world_name:=pf_test_world_4.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=True, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Basic_APF_Jackal_Kinova.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class Basic_APF_Jackal_Kinova_Rob(Basic_APF_Jackal_Kinova):
    real_robot = True


class Basic_APF_with_PD_Jackal_Kinova(APF_Env):
    def reset(self):
        return


class Clustered_APF_Jackal_Kinova(APF_Env):
    def reset(self):
        return


class Clustered_APF_with_PD_Jackal_Kinova(APF_Env):
    def reset(self):
        return
