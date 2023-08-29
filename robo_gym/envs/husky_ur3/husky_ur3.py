#!/usr/bin/env python3

import sys, time, math, copy
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from robo_gym.utils import utils, husky_ur3_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2


RS_TARGET = 0
RS_ROBOT_POSE = RS_TARGET + 3
RS_ROBOT_TWIST = RS_ROBOT_POSE + 3
RS_SCAN = RS_ROBOT_TWIST + 2
RS_COLLISION = RS_SCAN + 609
RS_OBSTACLES = RS_COLLISION + 1


class Husky_ur3_Env(gym.Env):
    """Mobile Industrial Robots Husky_ur3 base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.

    Attributes:
        Husky_ur3 (:obj:): Robot utilities object.
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
    laser_len = 609
    max_episode_steps = 512

    def __init__(self, rs_address=None, **kwargs):
        action_low = [0.0, -1.0]  # linear, angular

        self.Husky_ur3 = husky_ur3_utils.Husky_ur3()
        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()
        self.action_space = spaces.Box(low=np.array(action_low), high=np.full((2), 1.0), dtype=np.float32)
        self.seed()
        self.distance_threshold = 0.3
        self.min_target_dist = 1.0
        # Maximum linear velocity (m/s) of Husky
        max_lin_vel = 1.0
        # Maximum angular velocity (rad/s) of Husky
        max_ang_vel = 0.7
        self.max_vel = np.array([max_lin_vel, max_ang_vel])

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
        # Scale action
        rs_action = np.multiply(action, self.max_vel)
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
        scan = [0.0] * 609
        collision = False
        obstacles = [0.0] * 9
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
            start_pose = self.client.get_state_msg().state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3]
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
            x_target=rs_state[RS_TARGET],
            y_target=rs_state[RS_TARGET + 1],
            x_origin=rs_state[RS_ROBOT_POSE],
            y_origin=rs_state[RS_ROBOT_POSE + 1],
        )
        # Rotate origin of polar coordinates frame to be matching with robot frame and normalize to +/- pi
        polar_theta = utils.normalize_angle_rad(polar_theta - rs_state[RS_ROBOT_POSE + 2])

        # Get Laser scanners data542
        raw_laser_scan = rs_state[RS_SCAN : RS_SCAN + 609]

        # Downsampling of laser values by picking every n-th value
        if self.laser_len > 0:
            laser = utils.downsample_list_to_len(raw_laser_scan, self.laser_len)
            # Compose environment state
            state = np.concatenate((np.array([polar_r, polar_theta]), rs_state[RS_ROBOT_TWIST : RS_ROBOT_TWIST + 2], laser))
        else:
            # Compose environment state
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
        vel_tolerance = 0.1
        # Robot velocity range used to determine if there is an error in the sensor readings
        max_lin_vel = self.Husky_ur3.get_max_lin_vel() + vel_tolerance
        min_lin_vel = self.Husky_ur3.get_min_lin_vel() - vel_tolerance
        max_ang_vel = self.Husky_ur3.get_max_ang_vel() + vel_tolerance
        min_ang_vel = self.Husky_ur3.get_min_ang_vel() - vel_tolerance
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

        threshold = 0.05
        if min(rs_state[RS_SCAN : RS_SCAN + 609]) < threshold:
            return True
        else:
            return False


class NoObstacleNavigationHusky_ur3(Husky_ur3_Env):
    laser_len = 0

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
        base_reward = -50 * euclidean_dist_2d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        # Power used by the motors
        linear_power = abs(action[0] * 0.30)
        angular_power = abs(action[1] * 0.01)
        reward -= linear_power
        reward -= angular_power

        # End episode if robot is outside of boundary box
        if self._robot_outside_of_boundary_box(rs_state[3:5]):
            reward = -200.0
            done = True
            info["final_status"] = "out of boundary"

        # The episode terminates with success if the distance between the robot
        # and the target is less than the distance threshold.
        if euclidean_dist_2d < self.distance_threshold:
            reward = 200.0
            done = True
            info["final_status"] = "success"

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info["final_status"] = "max_steps_exceeded"

        return reward, done, info


class NoObstacleNavigationHusky_ur3_Sim(NoObstacleNavigationHusky_ur3, Simulation):
    cmd = "roslaunch husky_ur3_robot_server sim_robot_server.launch world_name:=empty_target_world.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        NoObstacleNavigationHusky_ur3.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class NoObstacleNavigationHusky_ur3_Rob(NoObstacleNavigationHusky_ur3):
    real_robot = True


class ObstacleAvoidanceHusky_ur3(Husky_ur3_Env):
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

        rs_state[RS_ROBOT_POSE : RS_ROBOT_POSE + 3] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose) == 3
        else:
            target_pose = self._get_target(start_pose)
        rs_state[RS_TARGET : RS_TARGET + 3] = target_pose

        # Generate obstacles positions
        self._generate_obstacles_positions()
        rs_state[RS_OBSTACLES : RS_OBSTACLES + 3] = self.sim_obstacles[0]
        rs_state[RS_OBSTACLES + 3 : RS_OBSTACLES + 6] = self.sim_obstacles[1]
        rs_state[RS_OBSTACLES + 6 : RS_OBSTACLES + 9] = self.sim_obstacles[2]

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
        target_coords = np.array([rs_state[RS_TARGET], rs_state[RS_TARGET + 1]])
        husky_coords = np.array([rs_state[RS_ROBOT_POSE], rs_state[RS_ROBOT_POSE + 1]])
        euclidean_dist_2d = np.linalg.norm(target_coords - husky_coords, axis=-1)
        # print("Target coordinate: {:.3f}, {:.3f}".format(rs_state[RS_TARGET], rs_state[RS_TARGET + 1]))
        # print("Distance: {:.3f}".format(euclidean_dist_2d))
        # print("Robot heading: {:.3f}".format(rs_state[RS_ROBOT_POSE + 2]))
        # print("Heading Error:: {:.3f}".format(self.state[1]))

        # Reward base
        base_reward = -90 * euclidean_dist_2d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        reward -= self.state[1]
        # Power used by the motors
        linear_power = abs(action[0] * 0.30)
        angular_power = abs(action[1] * 0.05)
        reward -= linear_power
        reward -= angular_power

        # End episode if robot is collides with an object, if it is too close
        # to an object.
        if not self.real_robot:
            if self._sim_robot_collision(rs_state):
                reward = -200.0
                done = True
                info["final_status"] = "collision"
                print("contact_collision")

            if self._min_laser_reading_below_threshold(rs_state):
                reward = -200
                done = True
                info["final_status"] = "collision"
                print("laser_collision")

            if self._robot_close_to_sim_obstacle(rs_state):
                reward = -200.0
                done = True
                info["final_status"] = "collision"
                print("obs_collision")

        if euclidean_dist_2d < self.distance_threshold:
            reward = 150
            done = True
            info["final_status"] = "success"

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info["final_status"] = "max_steps_exceeded"
            print("max_step_exceeded")

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
            x_t = self.np_random.uniform(low=-2.0, high=2.0)
            if robot_coordinates[1] > 0:
                y_t = self.np_random.uniform(low=-3.1, high=-2.1)
            else:
                y_t = self.np_random.uniform(low=2.1, high=3.1)
            yaw_t = 0.0
            target_dist = np.linalg.norm(np.array([x_t, y_t]) - np.array(robot_coordinates[0:2]), axis=-1)
            if target_dist >= self.min_target_dist:
                target_far_enough = True

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
        safety_radius = 0.38

        robot_close_to_obstacle = False
        robot_corners = self.Husky_ur3.get_corners_positions(rs_state[RS_ROBOT_POSE], rs_state[RS_ROBOT_POSE + 1], rs_state[RS_ROBOT_POSE + 2])

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

    def _generate_obstacles_positions(
        self,
    ):
        """Generate random positions for 3 obstacles.

        Used only for simulated Robot Server.

        """

        x_0 = self.np_random.uniform(low=-2.4, high=-1.9)
        y_0 = self.np_random.uniform(low=-1.0, high=1.0)
        yaw_0 = self.np_random.uniform(low=-np.pi, high=np.pi)
        x_1 = self.np_random.uniform(low=-0.4, high=0.4)
        y_1 = self.np_random.uniform(low=-1.0, high=1.0)
        yaw_1 = self.np_random.uniform(low=-np.pi, high=np.pi)
        x_2 = self.np_random.uniform(low=1.9, high=2.4)
        y_2 = self.np_random.uniform(low=-1.0, high=1.0)
        yaw_2 = self.np_random.uniform(low=-np.pi, high=np.pi)

        self.sim_obstacles = [[x_0, y_0, yaw_0], [x_1, y_1, yaw_1], [x_2, y_2, yaw_2]]


class ObstacleAvoidanceHusky_ur3_Sim(ObstacleAvoidanceHusky_ur3, Simulation):
    cmd = "roslaunch husky_ur3_robot_server sim_robot_server.launch world_name:=lab_6x8.world"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=True, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ObstacleAvoidanceHusky_ur3.__init__(self, rs_address=self.robot_server_ip, **kwargs)


class ObstacleAvoidanceHusky_ur3_Rob(ObstacleAvoidanceHusky_ur3):
    real_robot = True
