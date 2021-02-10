from copy import deepcopy
import math, copy
import numpy as np
from scipy.spatial.transform import Rotation as R
import gym
from gym import spaces
from gym.utils import seeding
from robo_gym.utils import utils, ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

from robo_gym.envs.ur5.ur5_avoid_B_moving_robot import ObstacleAvoidanceVarB1Box1PointUR5


# ? Variant C - 3 different target points that the robot should reach while staying as close as 
# ? possible to the original trajectory

DEBUG = False

class IrosEnv01UR5(ObstacleAvoidanceVarB1Box1PointUR5):
    def reset(self, initial_joint_positions = None, type='random'):
        """Environment reset.

        Args:
            initial_joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        # Initialize state machine variables
        self.state_n = 0 
        self.elapsed_steps_in_current_state = 0 
        self.target_reached = 0
        self.target_reached_counter = 0

        # Random 2
        self.r2 = np.random.uniform()

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        # NOTE: maybe we can find a cleaner version when we have the final envs (we could prob remove it for the avoidance task altogether)
        # Set initial robot joint positions
        if initial_joint_positions:
            assert len(initial_joint_positions) == 6
            self.initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            self.initial_joint_positions = self.last_position_on_success
        else:
            self.initial_joint_positions = self._get_desired_joint_positions()

        rs_state[6:12] = self.ur._ur_joint_list_to_ros_joint_list(self.initial_joint_positions)


        # TODO: We should create some kind of helper function depending on how dynamic these settings should be
        # Set initial state of the Robot Server
        n_sampling_points = int(np.random.default_rng().uniform(low= 4000, high=8000))
        
        string_params = {"object_0_function": "3d_spline"}

        r = np.random.uniform()

        if r <= 0.75:
            # object in front of the robot
            float_params = {"object_0_x_min": -0.7, "object_0_x_max": 0.7, "object_0_y_min": 0.2, "object_0_y_max": 1.5, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": n_sampling_points}
        elif r <= 0.81:
            # object behind robot
            float_params = {"object_0_x_min": -0.7, "object_0_x_max": 0.7, "object_0_y_min": - 1.5, "object_0_y_max": -0.2, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": n_sampling_points}
        elif r <= 0.87:
            # object on the left side of the  robot
            float_params = {"object_0_x_min": 0.3, "object_0_x_max": 1.5, "object_0_y_min": - 0.7, "object_0_y_max": 0.7, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": n_sampling_points}
        elif r <= 0.93:
            # object on top of the  robot
            float_params = {"object_0_x_min": -0.7, "object_0_x_max": 0.7, "object_0_y_min": - 0.7, "object_0_y_max": 0.7, \
                            "object_0_z_min": 0.6, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": n_sampling_points}
        else :
            # object on the right side of the  robot
            float_params = {"object_0_x_min": -0.2, "object_0_x_max": -1.5, "object_0_y_min": - 0.7, "object_0_y_max": 0.7, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": n_sampling_points}


        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))
        self.prev_rs_state = copy.deepcopy(rs_state)

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")
        
        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # save start position
        self.start_position = self.state[3:9]

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
        
        # check if current position is in the range of the initial joint positions
        if (len(self.last_position_on_success) == 0) or (type=='random'):
            joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
            if DEBUG:
                print("Initial Joint Positions")
                print(self.initial_joint_positions)
                print("Joint Positions")
                print(joint_positions)
            if not np.isclose(joint_positions, self.initial_joint_positions, atol=0.1).all():
                raise InvalidStateError('Reset joint positions are not within defined range')
            
        return self.state

    def step(self, action):
        self.elapsed_steps += 1
        self.elapsed_steps_in_current_state += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action = np.array(action)
        if self.last_action is None:
            self.last_action = action
        
        # Convert environment action to Robot Server action
        desired_joint_positions = copy.deepcopy(self._get_desired_joint_positions())
        if action.size == 3:
            desired_joint_positions[1:4] = desired_joint_positions[1:4] + action
        elif action.size == 5:
            desired_joint_positions[0:5] = desired_joint_positions[0:5] + action
        elif action.size == 6:
            desired_joint_positions = desired_joint_positions + action

        rs_action = desired_joint_positions

        # Convert action indexing from ur to ros
        rs_action = self.ur._ur_joint_list_to_ros_joint_list(rs_action)
        # Send action to Robot Server and get state
        rs_state = self.client.send_action_get_state(rs_action.tolist()).state
        self.prev_rs_state = copy.deepcopy(rs_state)

        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)
        
        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
        
        if DEBUG:
            print("Desired Joint Positions")
            print(self._get_desired_joint_positions())
            print("Joint Positions")
            print(self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12]))

        # Check if the robot is at the target position
        if self.target_point_flag:
            if np.isclose(self._get_desired_joint_positions(), self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12]), atol = 0.1).all():
                self.target_reached = 1
                self.state_n +=1
                # Restart from state 0 if the full trajectory has been completed
                self.state_n = self.state_n % len(TRAJECTORY)
                self.elapsed_steps_in_current_state = 0
                self.target_reached_counter += 1
        if DEBUG:
            print("Target Reached: ")
            print(self.target_reached)
            print("State number: ")
            print(self.state_n)
    
        # Assign reward
        reward = 0
        done = False
        reward, done, info = self._reward(rs_state=rs_state, action=action)
        self.last_action = action
        self.target_reached = 0

        return self.state, reward, done, info

    def print_state_action_info(self, rs_state, action):
        env_state = self._robot_server_state_to_env_state(rs_state)

        print('Action:', action)
        print('Last A:', self.last_action)
        print('Distance: {:.2f}'.format(env_state[0]))
        # print('Polar 1 (degree): {:.2f}'.format(env_state[1] * 180/math.pi))
        # print('Polar 2 (degree): {:.2f}'.format(env_state[2] * 180/math.pi))
        print('Joint Positions: [1]:{:.2f} [2]:{:.2f} [3]:{:.2f} [4]:{:.2f} [5]:{:.2f} [6]:{:.2f}'.format(*env_state[3:9]))
        print('Joint PosDeltas: [1]:{:.2f} [2]:{:.2f} [3]:{:.2f} [4]:{:.2f} [5]:{:.2f} [6]:{:.2f}'.format(*env_state[9:15]))
        print('Current Desired: [1]:{:.2f} [2]:{:.2f} [3]:{:.2f} [4]:{:.2f} [5]:{:.2f} [6]:{:.2f}'.format(*env_state[15:21]))
        print('Is current disred a target?', env_state[21])
        print('Targets reached', self.target_reached_counter)
        print('Sum of Deltas: {:.2f}'.format(sum(abs(env_state[9:15]))))
        print('Square of Deltas: {:.2f}'.format(np.square(env_state[9:15]).sum()))
        print()

    def print_reward_composition(self):
        # self.reward_composition.append([dr, act_r, small_actions, act_delta, dist_1, self.target_reached, collision_reward])
        dr = [r[0] for r in self.reward_composition]
        act_r = [r[1] for r in self.reward_composition]
        small_actions = [r[2] for r in self.reward_composition]
        act_delta = [r[3] for r in self.reward_composition]
        dist_1 = [r[4] for r in self.reward_composition]
        target_reached = [r[5] for r in self.reward_composition]
        collision_reward = [r[6] for r in self.reward_composition]

        print('Reward Composition of Episode:')
        print('Reward for keeping low delta joints: SUM={} MIN={}, MAX={}'.format(np.sum(dr), np.min(dr), np.max(dr)))
        print('Reward for as less as possible: SUM={} MIN={}, MAX={}'.format(np.sum(act_r), np.min(act_r), np.max(act_r)))
        print('Reward minor actions: SUM={} MIN={}, MAX={}'.format(np.sum(small_actions), np.min(small_actions), np.max(small_actions)))
        print('Punishment for rapid movement: SUM={} MIN={}, MAX={}'.format(np.sum(act_delta), np.min(act_delta), np.max(act_delta)))
        print('Punishment for target distance: SUM={} MIN={}, MAX={}'.format(np.sum(dist_1), np.min(dist_1), np.max(dist_1)))
        print('Reward for target reached: SUM={} MIN={}, MAX={}'.format(np.sum(target_reached), np.min(target_reached), np.max(target_reached)))
        print('Punishment for collision: SUM={} MIN={}, MAX={}'.format(np.sum(collision_reward), np.min(collision_reward), np.max(collision_reward)))
       
    # # semi working
    # def _reward(self, rs_state, action):
    #     # TODO: remove print when not needed anymore
    #     # print('action', action)
    #     env_state = self._robot_server_state_to_env_state(rs_state)

    #     reward = 0
    #     done = False
    #     info = {}

    #     # minimum and maximum distance the robot should keep to the obstacle
    #     minimum_distance = 0.45 # m
        
    #     distance_to_target = env_state[0]   
    #     delta_joint_pos = env_state[9:15]


    #     # reward for being in the defined interval of minimum_distance and maximum_distance
    #     dr = 0
    #     # if abs(delta_joint_pos).sum() < 0.5:
    #     #     dr = 1.5 * (1 - (sum(abs(delta_joint_pos))/0.5)) * (1/1000)
    #     #     reward += dr
    #     for delta in delta_joint_pos:
    #         if abs(delta) < 0.1:
    #             dr = 1.5 * (1 - (abs(delta))/0.1) * (1/1000)
    #             reward += dr
        
        
    #     # reward moving as less as possible
    #     act_r = 0 
    #     if abs(action).sum() <= action.size:
    #         act_r = 1.5 * (1 - (np.square(action).sum()/action.size)) * (1/1000)
    #         reward += act_r

    #     for a in action:
    #         if a < 0.1:
    #             reward += 0.1 * (1/1000)

        

    #     # punish big deltas in action
    #     act_delta = 0
    #     for i in range(len(action)):
    #         if abs(action[i] - self.last_action[i]) > 0.4:
    #             a_r = - 0.5 * (1/1000)
    #             act_delta += a_r
    #             reward += a_r
        
    #     dist_1 = 0
    #     if (distance_to_target < minimum_distance):
    #         dist_1 = -4 * (1/1000) # -2
    #         reward += dist_1

    #     if self.target_reached:
    #         reward += 0.05

    #     # TODO: we could remove this if we do not need to punish failure or reward success
    #     # Check if robot is in collision
    #     collision = True if rs_state[25] == 1 else False
    #     if collision:
    #         reward = -0.05
    #         done = True
    #         info['final_status'] = 'collision'

    #     if self.elapsed_steps >= self.max_episode_steps:
    #         done = True
    #         info['final_status'] = 'success'

        

    #     if DEBUG: self.print_state_action_info(rs_state, action)
    #     # ? DEBUG PRINT
    #     if DEBUG: print('reward composition:', 'dr =', round(dr, 5), 'no_act =', round(act_r, 5), 'min_dist_1 =', round(dist_1, 5), 'min_dist_2 =', 'delta_act', round(act_delta, 5))


    #     return reward, done, info

    # semi working
    def _reward(self, rs_state, action):
        # TODO: remove print when not needed anymore
        # print('action', action)
        env_state = self._robot_server_state_to_env_state(rs_state)

        reward = 0
        done = False
        info = {}

        # minimum and maximum distance the robot should keep to the obstacle
        minimum_distance = 0.45 # m
        
        distance_to_target = env_state[0]   
        delta_joint_pos = env_state[9:15]


        # reward for being in the defined interval of minimum_distance and maximum_distance
        dr = 0
        # if abs(delta_joint_pos).sum() < 0.5:
        #     dr = 1.5 * (1 - (sum(abs(delta_joint_pos))/0.5)) * (1/1000)
        #     reward += dr
        for delta in delta_joint_pos:
            if abs(delta) < 0.1:
                dr = 1.5 * (1 - (abs(delta))/0.1) * (1/1000) 
                dr = dr/5
                reward += dr
        
        
        # reward moving as less as possible
        act_r = 0 
        if abs(action).sum() <= action.size:
            act_r = 1.5 * (1 - (np.square(action).sum()/action.size)) * (1/1000)
            reward += act_r

        small_actions = 0
        for a in action:
            if a < 0.1:
                small_actions += 0.1 * (1/1000)
                reward += 0.1 * (1/1000)

        

        # punish big deltas in action
        act_delta = 0
        for i in range(len(action)):
            if abs(action[i] - self.last_action[i]) > 0.4:
                a_r = - 0.5 * (1/1000)
                act_delta += a_r
                reward += a_r
        
        dist_1 = 0
        if (distance_to_target < minimum_distance):
            dist_1 = -8 * (1/1000) # -2
            reward += dist_1

        tr_reward = 0
        if self.target_reached:
            tr_reward += 0.05
            reward += 0.05

        

        # TODO: we could remove this if we do not need to punish failure or reward success
        # Check if robot is in collision
        collision_reward = 0
        collision = True if rs_state[25] == 1 else False
        if collision:
            collision_reward = -0.5
            reward = - 0.5
            done = True
            info['final_status'] = 'collision'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
        

        self.reward_composition.append([dr, act_r, small_actions, act_delta, dist_1, tr_reward, collision_reward])
        if done:
            self.print_reward_composition()

        self.print_state_action_info(rs_state, action)
        # ? DEBUG PRINT
        if DEBUG: print('reward composition:', 'dr =', round(dr, 5), 'no_act =', round(act_r, 5), 'min_dist_1 =', round(dist_1, 5), 'min_dist_2 =', 'delta_act', round(act_delta, 5))


        return reward, done, info

    def _get_desired_joint_positions(self):
        """Get desired robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        if self.r2<=0.9:
            if self.elapsed_steps_in_current_state < len(TRAJECTORY[self.state_n]):
                joint_positions = copy.deepcopy(TRAJECTORY[self.state_n][self.elapsed_steps_in_current_state])
                self.target_point_flag = 0
            else:
                # Get last point of the trajectory segment
                joint_positions = copy.deepcopy(TRAJECTORY[self.state_n][-1])
                self.target_point_flag = 1
        else:
            # Get fixed joint positions
            joint_positions = np.array([-0.78,-1.31,-1.31,-2.18,1.57,0.0])
            self.target_point_flag = 0

        return joint_positions

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
        # with respect to the end effector frame
        target_coord = rs_state[0:3]
        
        ee_to_ref_frame_translation = np.array(rs_state[18:21])
        ee_to_ref_frame_quaternion = np.array(rs_state[21:25])
        ee_to_ref_frame_rotation = R.from_quat(ee_to_ref_frame_quaternion)
        ref_frame_to_ee_rotation = ee_to_ref_frame_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        ref_frame_to_ee_quaternion = ref_frame_to_ee_rotation.as_quat()
        # t' = - R^-1 * t
        ref_frame_to_ee_translation = -ref_frame_to_ee_rotation.apply(ee_to_ref_frame_translation)

        target_coord_ee_frame = utils.change_reference_frame(target_coord,ref_frame_to_ee_translation,ref_frame_to_ee_quaternion)
        target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)

        # Transform joint positions and joint velocities from ROS indexing to
        # standard indexing
        ur_j_pos = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
        ur_j_vel = self.ur._ros_joint_list_to_ur_joint_list(rs_state[12:18])

        # Normalize joint position values
        ur_j_pos_norm = self.ur.normalize_joint_values(joints=ur_j_pos)

        # desired joint positions
        desired_joints = self.ur.normalize_joint_values(self._get_desired_joint_positions())
        delta_joints = ur_j_pos_norm - desired_joints
        target_point_flag = self.target_point_flag

        # Compose environment state
        state = np.concatenate((target_polar, ur_j_pos_norm, delta_joints, desired_joints, [target_point_flag]))

        return state

    def _get_observation_space(self):
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """

        # Joint position range tolerance
        pos_tolerance = np.full(6,0.1)
        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(6, -1.0), pos_tolerance)
        # Target coordinates range
        target_range = np.full(3, np.inf)
        
        max_delta_start_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_delta_start_positions = np.subtract(np.full(6, -1.0), pos_tolerance)

        # Definition of environment observation_space
        max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions, max_joint_positions, [1]))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions, min_joint_positions, [0]))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

class IrosEnv01UR5DoF3(IrosEnv01UR5):
    def _get_action_space(self):
        return spaces.Box(low=np.full((3), -1.0), high=np.full((3), 1.0), dtype=np.float32)

class IrosEnv01UR5DoF5(IrosEnv01UR5):
    def _get_action_space(self):
        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)

class IrosEnv01UR5Sim(IrosEnv01UR5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=moving \
        n_objects:=1.0 \
        object_0_model_name:=box100 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv01UR5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv01UR5Rob(IrosEnv01UR5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"

class IrosEnv01UR5DoF3Sim(IrosEnv01UR5DoF3, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=moving \
        n_objects:=1.0 \
        object_0_model_name:=box100 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv01UR5DoF3.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv01UR5DoF3Rob(IrosEnv01UR5DoF3):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"

class IrosEnv01UR5DoF5Sim(IrosEnv01UR5DoF5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=moving \
        n_objects:=1.0 \
        object_0_model_name:=box100 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        IrosEnv01UR5DoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class IrosEnv01UR5DoF5Rob(IrosEnv01UR5DoF5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"


TRAJECTORY = [[[-1.23417836824526, -1.8541234175311487, -1.7814424673663538, -1.0691469351397913, 1.5949101448059082, 0], [-1.2341902891742151, -1.853800121937887, -1.777196232472555, -1.0738547484027308, 1.5948981046676636, 0], [-1.2341306845294397, -1.8532732168780726, -1.7704656759845179, -1.081209961568014, 1.594994068145752, 0], [-1.2341182867633265, -1.852626148854391, -1.7629073301898401, -1.0892837683307093, 1.5951018333435059, 0], [-1.2340944449054163, -1.8519070784198206, -1.753885571156637, -1.0990708510028284, 1.594946026802063, 0], [-1.2340224424945276, -1.8510573546039026, -1.7434843222247522, -1.1104868094073694, 1.5949820280075073, 0], [-1.2339866797076624, -1.8498953024493616, -1.729339901600973, -1.125544850026266, 1.594946026802063, 0], [-1.2339265982257288, -1.8488052527057093, -1.715663258229391, -1.1404474417315882, 1.5951018333435059, 0], [-1.2338789145099085, -1.8474996725665491, -1.700559441243307, -1.1567757765399378, 1.595065951347351, 0], [-1.2338064352618616, -1.8459184805499476, -1.6809924284564417, -1.1779444853412073, 1.59516179561615, 0], [-1.2337225119220179, -1.8444932142840784, -1.66254169145693, -1.1979029814349573, 1.5949820280075073, 0], [-1.2336629072772425, -1.842923943196432, -1.6429513136493128, -1.219276253377096, 1.59516179561615, 0], [-1.2335789839373987, -1.84094745317568, -1.6177104155169886, -1.2464245001422327, 1.5950778722763062, 0], [-1.233495060597555, -1.839175049458639, -1.5947015921222132, -1.2712367216693323, 1.5951378345489502, 0], [-1.2333868185626429, -1.837318245564596, -1.5714286009417933, -1.2966378370868128, 1.595005989074707, 0], [-1.2333267370807093, -1.8354375998126429, -1.5481675306903284, -1.3216307798968714, 1.5951257944107056, 0], [-1.2332428137408655, -1.8332341353045862, -1.5209487120257776, -1.3509624640094202, 1.5951378345489502, 0], [-1.2331112066852015, -1.8314250151263636, -1.4976409117328089, -1.3759210745440882, 1.595221757888794, 0], [-1.2330511252032679, -1.8296640555011194, -1.474440876637594, -1.4011443297015589, 1.595233678817749, 0], [-1.2329672018634241, -1.827364746724264, -1.4473660627948206, -1.43040639558901, 1.5952576398849487, 0], [-1.2328713575946253, -1.8255322615252894, -1.4240229765521448, -1.455594841633932, 1.5952457189559937, 0], [-1.232751194630758, -1.8236992994891565, -1.400679890309469, -1.480783764516012, 1.5951497554779053, 0], [-1.2326791922198694, -1.82151967683901, -1.3734267393695276, -1.5100720564471644, 1.59516179561615, 0], [-1.2326553503619593, -1.8195670286761683, -1.3500965277301233, -1.5352862516986292, 1.595233678817749, 0], [-1.2325714270221155, -1.8177340666400355, -1.326886002217428, -1.5602377096759241, 1.595293641090393, 0], [-1.2324636618243616, -1.8159502188311976, -1.30353290239443, -1.5855138937579554, 1.5952576398849487, 0], [-1.2323558966266077, -1.813782040272848, -1.2762931028949183, -1.6147688070880335, 1.5954374074935913, 0], [-1.2322595755206507, -1.8119853178607386, -1.2530844847308558, -1.6400216261493128, 1.595233678817749, 0], [-1.2322233358966272, -1.8101046721087855, -1.2305954138385218, -1.6643279234515589, 1.595353603363037, 0], [-1.232079807912008, -1.8081286589251917, -1.206259552632467, -1.6904452482806605, 1.5952576398849487, 0], [-1.2320435682879847, -1.8066547552691858, -1.1869738737689417, -1.7113130728351038, 1.5954134464263916, 0], [-1.2319958845721644, -1.8052776495562952, -1.1693313757525843, -1.7305153051959437, 1.595353603363037, 0], [-1.2319119612323206, -1.803732697163717, -1.1506574789630335, -1.7507002989398401, 1.5953655242919922, 0], [-1.2318757216082972, -1.8026307264911097, -1.136266056691305, -1.7664273420916956, 1.5953775644302368, 0], [-1.231839958821432, -1.801600758229391, -1.1230610052691858, -1.7803800741778772, 1.5953415632247925, 0], [-1.2316840330706995, -1.8006780783282679, -1.111368481312887, -1.7933266798602503, 1.5954853296279907, 0], [-1.2310603300677698, -1.7996004263507288, -1.0985358397113245, -1.8070038000689905, 1.5953655242919922, 0], [-1.230196777974264, -1.7986181418048304, -1.0880778471576136, -1.8183682600604456, 1.5954374074935913, 0], [-1.2287338415728968, -1.7976720968829554, -1.0781243483172815, -1.8293970266925257, 1.5954134464263916, 0], [-1.226107422505514, -1.796450440083639, -1.066827122365133, -1.841708008443014, 1.5954374074935913, 0], [-1.2228811422931116, -1.7953484694110315, -1.0577967802630823, -1.851814095173971, 1.595389485359192, 0], [-1.218647305165426, -1.7942102591144007, -1.0496066252337855, -1.8612001577960413, 1.595221757888794, 0], [-1.212243382130758, -1.7928808371173304, -1.0410917440997522, -1.8710301558123987, 1.5952097177505493, 0], [-1.205622975026266, -1.7916353384601038, -1.034975830708639, -1.8783071676837366, 1.5951018333435059, 0], [-1.1976593176471155, -1.7903655211078089, -1.0301788488971155, -1.884289566670553, 1.5951257944107056, 0], [-1.1887968222247522, -1.7891200224505823, -1.0269048849688929, -1.8888686339007776, 1.5948143005371094, 0], [-1.1771157423602503, -1.7876227537738245, -1.0247939268695276, -1.8923338095294397, 1.594838261604309, 0], [-1.1661542097674769, -1.7863653341876429, -1.0245545546161097, -1.8938444296466272, 1.5946824550628662, 0], [-1.1544368902789515, -1.785119358693258, -1.0253217856036585, -1.894167725239889, 1.5944180488586426, 0], [-1.1405490080462855, -1.7836464087115687, -1.026916805897848, -1.8939521948443812, 1.594406008720398, 0], [-1.1275365988360804, -1.7823527495013636, -1.028355900441305, -1.893832031880514, 1.5941663980484009, 0], [-1.1128571669207972, -1.7808674017535608, -1.0301907698260706, -1.893496338521139, 1.5939147472381592, 0], [-1.0938733259784144, -1.7789271513568323, -1.0325053373919886, -1.8931248823748987, 1.593578815460205, 0], [-1.0757277647601526, -1.7771185080157679, -1.0347841421710413, -1.8927291075335901, 1.5934709310531616, 0], [-1.056023899708883, -1.7750704924212855, -1.037086311970846, -1.8921778837787073, 1.5931113958358765, 0], [-1.0347369352923792, -1.7729862372027796, -1.0396884123431605, -1.8916857878314417, 1.5930155515670776, 0], [-1.0077179113971155, -1.7701595465289515, -1.0428670088397425, -1.89101487794985, 1.5926315784454346, 0], [-0.9831808249102991, -1.7676447073565882, -1.0459607283221644, -1.8904393354998987, 1.5923200845718384, 0], [-0.9581406752215784, -1.7651169935809534, -1.048898998891012, -1.8899357954608362, 1.5920683145523071, 0], [-0.9287594000445765, -1.762111012135641, -1.052544418965475, -1.8891571203814905, 1.5914208889007568, 0], [-0.9036958853351038, -1.7594640890704554, -1.0555904547320765, -1.8885687033282679, 1.59120512008667, 0], [-0.8786442915545862, -1.75684100786318, -1.0586965719806116, -1.8879697958575647, 1.5907613039016724, 0], [-0.853401009236471, -1.7542780081378382, -1.0617182890521448, -1.8873584906207483, 1.5906175374984741, 0], [-0.8240931669818323, -1.751366917287008, -1.0653999487506312, -1.886771027241842, 1.5902456045150757, 0], [-0.7989819685565394, -1.748744312916891, -1.0683859030352991, -1.886087719594137, 1.5897423028945923, 0], [-0.7738598028766077, -1.7461455504046839, -1.0714200178729456, -1.885547939931051, 1.5895745754241943, 0], [-0.7445529142962855, -1.7431028524981897, -1.074874226246969, -1.8848407904254358, 1.589106798171997, 0], [-0.7192266623126429, -1.7406237761126917, -1.077860180531637, -1.8842294851886194, 1.5890109539031982, 0], [-0.6940696875201624, -1.7380364576922815, -1.0808947722064417, -1.883749787007467, 1.588531494140625, 0], [-0.6646917502032679, -1.7350066343890589, -1.0844085852252405, -1.8829949537860315, 1.588039755821228, 0], [-0.6395943800555628, -1.7323959509478968, -1.0874541441546839, -1.8824675718890589, 1.5875720977783203, 0], [-0.6144622007953089, -1.729868237172262, -1.0906084219561976, -1.8818915526019495, 1.587380290031433, 0], [-0.5894020239459437, -1.727292839680807, -1.0936549345599573, -1.881148640309469, 1.5872005224227905, 0], [-0.560157601033346, -1.724346939717428, -1.09727651277651, -1.8804648558246058, 1.5868409872055054, 0], [-0.5352180639850062, -1.7216876188861292, -1.100274387990133, -1.8799856344806116, 1.5863373279571533, 0], [-0.5101230780230921, -1.719172779713766, -1.1032732168780726, -1.8793981711017054, 1.586013674736023, 0], [-0.4807956854449671, -1.7162025610553187, -1.1068108717547815, -1.8787506262408655, 1.5855215787887573, 0], [-0.455665413533346, -1.7135918776141565, -1.109856907521383, -1.8781393210040491, 1.58530592918396, 0], [-0.430427376423971, -1.711052719746725, -1.112927261983053, -1.877456013356344, 1.5851022005081177, 0], [-0.40124589601625615, -1.7080219427691858, -1.116464916859762, -1.8768327871905726, 1.5844546556472778, 0], [-0.3761404196368616, -1.7054107824908655, -1.1195595900165003, -1.8764012495623987, 1.584298849105835, 0], [-0.35100013414491826, -1.7028840223895472, -1.122617546712057, -1.8757179419146937, 1.5838310718536377, 0], [-0.32583600679506475, -1.7002971808062952, -1.125699822102682, -1.8749983946429651, 1.5836033821105957, 0], [-0.29664403596986944, -1.6972668806659144, -1.1292737166034144, -1.87429124513735, 1.583124041557312, 0], [-0.271336857472555, -1.6946199576007288, -1.1322358290301722, -1.8737996260272425, 1.582859992980957, 0], [-0.24618608156313115, -1.6920693556415003, -1.1352341810809534, -1.87329608598818, 1.5824165344238281, 0], [-0.21676808992494756, -1.6891706625567835, -1.1387961546527308, -1.8725770155536097, 1.5822128057479858, 0], [-0.19166595140566045, -1.6866076628314417, -1.141794506703512, -1.8718932310687464, 1.5817691087722778, 0], [-0.16649276414980108, -1.6839607397662562, -1.1448290983783167, -1.8714378515826624, 1.5815293788909912, 0], [-0.1372922102557581, -1.6810148398028772, -1.1483309904681605, -1.870587174092428, 1.5810256004333496, 0], [-0.11209565797914678, -1.6784876028644007, -1.1514728705035608, -1.870035473500387, 1.5807020664215088, 0], [-0.08703166643251592, -1.6759126822101038, -1.1545794645892542, -1.8694241682635706, 1.580282211303711, 0], [-0.06206351915468389, -1.6733735243426722, -1.1575658957110804, -1.8688367048846644, 1.5801024436950684, 0], [-0.03499919572939092, -1.6704872290240687, -1.1608403364764612, -1.8681767622577112, 1.5797070264816284, 0], [-0.01385528246034795, -1.6683195273028772, -1.1635144392596644, -1.8675897757159632, 1.5794070959091187, 0], [0.006304451264441013, -1.6662958304034632, -1.1658766905414026, -1.8671820799456995, 1.5790235996246338, 0], [0.027423277497291565, -1.6640318075763147, -1.1683958212481897, -1.8667987028705042, 1.5787838697433472, 0], [0.043986976146698, -1.6624143759356897, -1.1704466978656214, -1.8663910071002405, 1.5786999464035034, 0], [0.05880097672343254, -1.660881821309225, -1.1722338835345667, -1.8661392370807093, 1.578376054763794, 0], [0.07413029670715332, -1.6593011061297815, -1.1740205923663538, -1.8657320181476038, 1.5781363248825073, 0], [0.08651075512170792, -1.6580551306353968, -1.1756518522845667, -1.8652165571795862, 1.5779565572738647, 0], [0.09860358387231827, -1.6569173971759241, -1.1775949637042444, -1.8644369284259241, 1.5777409076690674, 0], [0.11006125062704086, -1.6559713522540491, -1.1799457708941858, -1.8628786245929163, 1.577596664428711, 0], [0.12230950593948364, -1.6551454702960413, -1.1837237516986292, -1.859917465840475, 1.5773450136184692, 0], [0.13172948360443115, -1.6546896139727991, -1.1877654234515589, -1.856213394795553, 1.577368974685669, 0], [0.1400708705186844, -1.6547377745257776, -1.1927550474749964, -1.851382080708639, 1.577081322669983, 0], [0.14842388033866882, -1.654905144368307, -1.1995909849749964, -1.8441293875323694, 1.5770573616027832, 0], [0.1542242467403412, -1.655360523854391, -1.2064040342914026, -1.836649243031637, 1.576853632926941, 0], [0.15891008079051971, -1.6561153570758265, -1.214283291493551, -1.8279097715960901, 1.5767334699630737, 0], [0.16246938705444336, -1.6573007742511194, -1.2231233755694788, -1.8179362455951136, 1.5768296718597412, 0], [0.1654658168554306, -1.658989731465475, -1.2343371550189417, -1.805122200642721, 1.576661467552185, 0], [0.16714361310005188, -1.6607978979693812, -1.2445204893695276, -1.793062988911764, 1.57664954662323, 0], [0.16811396181583405, -1.6627982298480433, -1.2555306593524378, -1.7801044623004358, 1.5765057802200317, 0], [0.1684255450963974, -1.6652053038226526, -1.2686522642718714, -1.7646773497210901, 1.576493740081787, 0], [0.16847386956214905, -1.6673129240619105, -1.2802274862872522, -1.7508085409747522, 1.5765177011489868, 0], [0.1684379130601883, -1.6695287863360804, -1.2921860853778284, -1.736880127583639, 1.5765776634216309, 0], [0.16841356456279755, -1.6723435560809534, -1.3074305693255823, -1.7185171286212366, 1.5765776634216309, 0], [0.16837762296199799, -1.6750739256488245, -1.3221843878375452, -1.701268498097555, 1.5763379335403442, 0], [0.16834166646003723, -1.6779606978045862, -1.3379815260516565, -1.6822827498065394, 1.5763739347457886, 0], [0.16826975345611572, -1.6811583677874964, -1.3554461638080042, -1.6616795698748987, 1.5763739347457886, 0], [0.16820983588695526, -1.6852424780475062, -1.3773368040667933, -1.6356352011310022, 1.5763139724731445, 0], [0.1680780053138733, -1.689099136983053, -1.3976452986346644, -1.6117361227618616, 1.5761462450027466, 0], [0.16801808774471283, -1.6930869261371058, -1.4187091032611292, -1.5864484945880335, 1.5761821269989014, 0], [0.1678866297006607, -1.6976621786700647, -1.4434316794024866, -1.5570977369891565, 1.5760383605957031, 0], [0.16780275106430054, -1.7017219702350062, -1.4645441214190882, -1.5319307486163538, 1.5761702060699463, 0], [0.16769489645957947, -1.7056391874896448, -1.4857648054706019, -1.5068123976336878, 1.5760263204574585, 0], [0.1675390899181366, -1.710034195576803, -1.5105121771441858, -1.477584187184469, 1.5760623216629028, 0], [0.16750314831733704, -1.7140229384051722, -1.5316250959979456, -1.4524195829974573, 1.5759540796279907, 0], [0.16741925477981567, -1.7180231253253382, -1.5527861754046839, -1.4273985067950647, 1.5759540796279907, 0], [0.1673114001750946, -1.7219632307635706, -1.5740678946124476, -1.4021027723895472, 1.5759061574935913, 0], [0.1672155261039734, -1.7266348044024866, -1.5986602942096155, -1.3727934996234339, 1.5759180784225464, 0], [0.16709567606449127, -1.730682675038473, -1.6199176947223108, -1.3476794401751917, 1.5757979154586792, 0], [0.16697584092617035, -1.734551254902975, -1.6412237326251429, -1.3225653807269495, 1.5756301879882812, 0], [0.16685599088668823, -1.739030663167135, -1.6658290068255823, -1.293330494557516, 1.5756421089172363, 0], [0.16678409278392792, -1.7429116407977503, -1.68695575395693, -1.2680266539203089, 1.5754863023757935, 0], [0.1667121797800064, -1.7469595114337366, -1.7082975546466272, -1.2430103460894983, 1.5755462646484375, 0], [0.1666039526462555, -1.751571003590719, -1.7331550757037562, -1.213489834462301, 1.5753304958343506, 0], [0.16649609804153442, -1.755547348652975, -1.7542455832110804, -1.188343350087301, 1.5754023790359497, 0], [0.1664002239704132, -1.7595236937152308, -1.775384251271383, -1.1633408705340784, 1.5754504203796387, 0], [0.16631633043289185, -1.7634642759906214, -1.796655003224508, -1.13823110262026, 1.5752466917037964, 0], [0.16617251932621002, -1.76799184480776, -1.8214290777789515, -1.1089418570147913, 1.575366497039795, 0], [0.16610099375247955, -1.7718246618853968, -1.842447582875387, -1.0837491194354456, 1.5751746892929077, 0], [0.1660171002149582, -1.7756574789630335, -1.8627951780902308, -1.0598638693438929, 1.575138807296753, 0], [0.16589725017547607, -1.779741112385885, -1.8847020308123987, -1.0338218847857874, 1.5749949216842651, 0], [0.16587328910827637, -1.7829516569720667, -1.9018457571612757, -1.0135071913348597, 1.575042963027954, 0], [0.16580137610435486, -1.7859342733966272, -1.917694393788473, -0.9944732824908655, 1.5751148462295532, 0], [0.1657174974679947, -1.789180103932516, -1.9343941847430628, -0.9744222799884241, 1.5749709606170654, 0], [0.1656935214996338, -1.7915752569781702, -1.9474595228778284, -0.9591863791095179, 1.5750188827514648, 0], [0.16566956043243408, -1.793779198323385, -1.959132496510641, -0.9455679098712366, 1.5749590396881104, 0], [0.16559764742851257, -1.7955763975726526, -1.9693182150470179, -0.9333146254168909, 1.5748507976531982, 0], [0.16552574932575226, -1.7974804083453577, -1.9796479384051722, -0.9211934248553675, 1.5749709606170654, 0], [0.16547781229019165, -1.7988575140582483, -1.9869540373431605, -0.9124382177935999, 1.5747908353805542, 0], [0.1654418557882309, -1.799959961568014, -1.9928324858294886, -0.905288044606344, 1.5748987197875977, 0], [0.16545383632183075, -1.800906006489889, -1.9981954733477991, -0.8991912047015589, 1.5747908353805542, 0], [0.16541787981987, -1.801504913960592, -2.001134697590963, -0.8955743948565882, 1.574886679649353, 0], [0.16541787981987, -1.8016966024981897, -2.0029099623309534, -0.8936217466937464, 1.5749590396881104, 0], [0.16541787981987, -1.8017085234271448, -2.0031264464007776, -0.893310848866598, 1.5748627185821533, 0], [0.16540589928627014, -1.8017085234271448, -2.0031140486346644, -0.893322769795553, 1.574886679649353, 0], [0.1653699427843094, -1.8017085234271448, -2.0031026045428675, -0.8933346907245081, 1.574874758720398, 0], [0.1653699427843094, -1.8016966024981897, -2.0031140486346644, -0.893310848866598, 1.5748627185821533, 0], [0.16538193821907043, -1.8017085234271448, -2.003174130116598, -0.8933346907245081, 1.574874758720398, 0], [0.16535796225070953, -1.8017204443561, -2.0031865278827112, -0.893310848866598, 1.5748627185821533, 0]], [[0.1654897928237915, -1.7891557852374476, -1.9727376143084925, -0.9362252394305628, 1.575042963027954, 0], [0.1655377298593521, -1.7853949705706995, -1.9639313856707972, -0.9489930311786097, 1.5749469995498657, 0], [0.16557368636131287, -1.7804482618915003, -1.9520543257342737, -0.9654987494098108, 1.5751746892929077, 0], [0.16564558446407318, -1.7758129278766077, -1.9406450430499476, -0.9817407766925257, 1.5750548839569092, 0], [0.16568154096603394, -1.7707226912127894, -1.9283354918109339, -0.9993365446673792, 1.5751028060913086, 0], [0.16577741503715515, -1.7640627066241663, -1.9122231642352503, -1.0219634214984339, 1.5753185749053955, 0], [0.1658373326063156, -1.7577388922320765, -1.8973830381976526, -1.0431769529925745, 1.5752825736999512, 0], [0.16594518721103668, -1.750972096120016, -1.8811867872821253, -1.065937344227926, 1.575498342514038, 0], [0.16602908074855804, -1.7425525824176233, -1.8611157576190394, -1.0943148771869105, 1.5754863023757935, 0], [0.1660889983177185, -1.7352097670184534, -1.8432634512530726, -1.1196630636798304, 1.57572603225708, 0], [0.16623243689537048, -1.7277839819537562, -1.8256281057940882, -1.1448081175433558, 1.5758339166641235, 0], [0.166304349899292, -1.7191370169269007, -1.804800812398092, -1.1742308775531214, 1.5758697986602783, 0], [0.1664002239704132, -1.7117827574359339, -1.7870333830462855, -1.1994136015521448, 1.5759661197662354, 0], [0.16649609804153442, -1.7044890562640589, -1.7694457213031214, -1.2245238463031214, 1.5759661197662354, 0], [0.16656799614429474, -1.6957462469684046, -1.7486313025103968, -1.2538054625140589, 1.5762420892715454, 0], [0.1666642427444458, -1.688404385243551, -1.7309477964984339, -1.2788813749896448, 1.576206088066101, 0], [0.16676011681556702, -1.6810029188739222, -1.7130959669696253, -1.304066006337301, 1.5763499736785889, 0], [0.16682003438472748, -1.6724990049945276, -1.69250995317568, -1.3332651297198694, 1.5763858556747437, 0], [0.1669159084558487, -1.6649659315692347, -1.6748388449298304, -1.3586066404925745, 1.5765776634216309, 0], [0.1669878214597702, -1.6576240698443812, -1.657191578541891, -1.3831465880023401, 1.576721429824829, 0], [0.16710765659809113, -1.6498635450946253, -1.6384766737567347, -1.4099515120135706, 1.5766735076904297, 0], [0.16716758906841278, -1.6436355749713343, -1.6234090963946741, -1.4310172239886683, 1.57692551612854, 0], [0.16720353066921234, -1.6378868261920374, -1.6096370855914515, -1.4507177511798304, 1.5769973993301392, 0], [0.1672874242067337, -1.6316826979266565, -1.5948694388019007, -1.471424404774801, 1.5770694017410278, 0], [0.1673114001750946, -1.6269763151751917, -1.5836165587054651, -1.4875906149493616, 1.577153205871582, 0], [0.16739527881145477, -1.6227605978595179, -1.5734198729144495, -1.50201923051943, 1.5770214796066284, 0], [0.16741925477981567, -1.6183293501483362, -1.5628388563739222, -1.5169747511493128, 1.5772491693496704, 0], [0.16746719181537628, -1.6152156035052698, -1.5552094618426722, -1.5279524962054651, 1.5772850513458252, 0], [0.16747917234897614, -1.6124852339373987, -1.5486114660846155, -1.5373595396624964, 1.5771892070770264, 0], [0.1675390899181366, -1.6097429434405726, -1.5422776381122034, -1.5463359991656702, 1.5773569345474243, 0], [0.16752710938453674, -1.6079943815814417, -1.5380428473102015, -1.5520766417132776, 1.5772371292114258, 0], [0.16759902238845825, -1.6068204084979456, -1.5349963347064417, -1.5563910643206995, 1.5772970914840698, 0], [0.16762298345565796, -1.605933968220846, -1.5329087416278284, -1.5593393484698694, 1.5772850513458252, 0], [0.16762298345565796, -1.6057665983783167, -1.532405201588766, -1.5601180235492151, 1.5774528980255127, 0], [0.16849783062934875, -1.6057308355914515, -1.5326326529132288, -1.5599859396563929, 1.5773329734802246, 0], [0.17138603329658508, -1.6055510679828089, -1.5330889860736292, -1.5597108046161097, 1.5773210525512695, 0], [0.1756160855293274, -1.6052277723895472, -1.5337842146502894, -1.5593393484698694, 1.5773329734802246, 0], [0.18157261610031128, -1.6046889464007776, -1.5347684065448206, -1.558847729359762, 1.577428936958313, 0], [0.1904166042804718, -1.6040061155902308, -1.536351505910055, -1.558176342641012, 1.577201247215271, 0], [0.19989612698554993, -1.6033113638507288, -1.537983242665426, -1.5571935812579554, 1.577309012413025, 0], [0.210837721824646, -1.6025689283954065, -1.539830509816305, -1.556019131337301, 1.5771771669387817, 0], [0.22572138905525208, -1.6015508810626429, -1.542337719594137, -1.5545809904681605, 1.577201247215271, 0], [0.24016200006008148, -1.6004489103900355, -1.5448210875140589, -1.5530951658831995, 1.5771173238754272, 0], [0.2563036382198334, -1.5992515722857874, -1.5475199858294886, -1.5515010992633265, 1.576913595199585, 0], [0.27696293592453003, -1.597778622304098, -1.5510948340045374, -1.5495360533343714, 1.5769015550613403, 0], [0.29654404520988464, -1.5963895956622522, -1.5544174353228968, -1.547678295766012, 1.5766255855560303, 0], [0.3177778422832489, -1.5948923269854944, -1.5580766836749476, -1.5456169287311, 1.5766375064849854, 0], [0.3444756865501404, -1.592952076588766, -1.562803093587057, -1.54301625887026, 1.5763858556747437, 0], [0.3691598176956177, -1.591179672871725, -1.5669897238360804, -1.540727440510885, 1.576421856880188, 0], [0.3944064676761627, -1.5894072691546839, -1.5712726751910608, -1.5381863752948206, 1.5762900114059448, 0], [0.42358317971229553, -1.5873354117022913, -1.5763347784625452, -1.5353100935565394, 1.5762181282043457, 0], [0.4487931430339813, -1.585442845021383, -1.5805695692645472, -1.5329497496234339, 1.5759180784225464, 0], [0.47385889291763306, -1.5836589972125452, -1.5848405996905726, -1.5306008497821253, 1.5758339166641235, 0], [0.5031895041465759, -1.5816105047809046, -1.5898788611041468, -1.5277965704547327, 1.5756421089172363, 0], [0.528398334980011, -1.579766575490133, -1.594233814870016, -1.5253995100604456, 1.5755462646484375, 0], [0.5534502267837524, -1.5779345671283167, -1.5985282103167933, -1.5229905287372034, 1.5755462646484375, 0], [0.5828752517700195, -1.575886074696676, -1.603506867085592, -1.5202224890338343, 1.575270652770996, 0], [0.6080226898193359, -1.5740421454059046, -1.6078255812274378, -1.5177777449237269, 1.5752586126327515, 0], [0.6331453323364258, -1.5721495787249964, -1.6121323744403284, -1.5153568426715296, 1.5750788450241089, 0], [0.662412703037262, -1.5700300375567835, -1.6171229521380823, -1.5124691168414515, 1.5749709606170654, 0], [0.6916197538375854, -1.5679100195514124, -1.6221378485309046, -1.509688679371969, 1.574658989906311, 0], [0.7170165777206421, -1.5661018530475062, -1.626575771962301, -1.5072558561908167, 1.574718952178955, 0], [0.7463183999061584, -1.5640175978290003, -1.6315787474261683, -1.504439655934469, 1.5744673013687134, 0], [0.7716545462608337, -1.562317196522848, -1.6358373800860804, -1.5020793120013636, 1.5745272636413574, 0], [0.7967625856399536, -1.5604375044452112, -1.6401446501361292, -1.499730412160055, 1.5742754936218262, 0], [0.8260868787765503, -1.5583294073687952, -1.6450989882098597, -1.4968541304217737, 1.5741437673568726, 0], [0.8511942028999329, -1.5565093199359339, -1.649393383656637, -1.4944332281695765, 1.573939561843872, 0], [0.8762891888618469, -1.5547602812396448, -1.6536524931537073, -1.491976563130514, 1.5737597942352295, 0], [0.9056363105773926, -1.5526288191424769, -1.6587031523333948, -1.4892204443561, 1.5737837553024292, 0], [0.9307425022125244, -1.5508444944964808, -1.6631177107440394, -1.4868353048907679, 1.5735201835632324, 0], [0.9559558033943176, -1.5490239302264612, -1.66742450395693, -1.4844144026385706, 1.573352336883545, 0], [0.981049656867981, -1.5471556822406214, -1.6716955343829554, -1.48204213777651, 1.573352336883545, 0], [1.0104424953460693, -1.5450480620013636, -1.6767342726336878, -1.4791901747332972, 1.5730044841766357, 0], [1.035535216331482, -1.5432637373553675, -1.6809924284564417, -1.4767454306231897, 1.5730763673782349, 0], [1.06061589717865, -1.5414193312274378, -1.6852753798114222, -1.4742768446551722, 1.5730044841766357, 0], [1.0899600982666016, -1.539323631917135, -1.6903985182391565, -1.4714725653277796, 1.5728965997695923, 0], [1.1151235103607178, -1.5375273863421839, -1.6946690718280237, -1.4690521399127405, 1.5726808309555054, 0], [1.1401551961898804, -1.5356829802142542, -1.6989520231830042, -1.4667509237872522, 1.5726089477539062, 0], [1.1692829132080078, -1.5335753599749964, -1.7039907614337366, -1.4638989607440394, 1.5724172592163086, 0], [1.1940984725952148, -1.5318625609027308, -1.7081778685199183, -1.4615863005267542, 1.572465181350708, 0], [1.2173807621002197, -1.5301983992206019, -1.712172810231344, -1.4593213240252894, 1.5721172094345093, 0], [1.242363452911377, -1.528365437184469, -1.7164433638202112, -1.4569366613971155, 1.5721172094345093, 0], [1.2621842622756958, -1.5269644896136683, -1.7198742071734827, -1.4550078550921839, 1.5719374418258667, 0], [1.2803523540496826, -1.525658909474508, -1.7229941526996058, -1.4532459417926233, 1.5717697143554688, 0], [1.299370527267456, -1.5242818037616175, -1.7263291517840784, -1.4514363447772425, 1.5716378688812256, 0], [1.3140411376953125, -1.5231922308551233, -1.728800121937887, -1.44999868074526, 1.5715539455413818, 0], [1.3270589113235474, -1.5222581068622034, -1.7310674826251429, -1.4487288633929651, 1.5717337131500244, 0], [1.3401966094970703, -1.5213244597064417, -1.7332389990436, -1.4475658575641077, 1.5715539455413818, 0], [1.3496934175491333, -1.520665470753805, -1.7349188963519495, -1.4466193358050745, 1.571578025817871, 0], [1.3577052354812622, -1.5200665632831019, -1.7362740675555628, -1.4459603468524378, 1.5714941024780273, 0], [1.3650107383728027, -1.5196235815631312, -1.737497631703512, -1.4451935927020472, 1.5714820623397827, 0], [1.3693939447402954, -1.5193598906146448, -1.7382176558123987, -1.4448702971087855, 1.5713741779327393, 0], [1.37230384349823, -1.5192282835589808, -1.7386615912066858, -1.4446900526629847, 1.5712419748306274, 0], [1.3735854625701904, -1.5191085974322718, -1.7387812773333948, -1.4445579687701624, 1.5712300539016724, 0], [1.3735015392303467, -1.519252125416891, -1.7392495314227503, -1.443719212208883, 1.571254014968872, 0], [1.3735015392303467, -1.5198386351214808, -1.7407129446612757, -1.4413707892047327, 1.5713379383087158, 0], [1.3734416961669922, -1.521479908620016, -1.7436884085284632, -1.436828915272848, 1.5712779760360718, 0], [1.3734296560287476, -1.5236113707171839, -1.7473114172564905, -1.431113068257467, 1.5714222192764282, 0], [1.3734056949615479, -1.5264022986041468, -1.7520020643817347, -1.4238155523883265, 1.5712779760360718, 0], [1.3733936548233032, -1.530353848134176, -1.7586963812457483, -1.4131749312030237, 1.5713379383087158, 0], [1.3732622861862183, -1.5344136396991175, -1.7655108610736292, -1.4022825399981897, 1.5712060928344727, 0], [1.3732023239135742, -1.5390241781817835, -1.7733209768878382, -1.38991624513735, 1.571254014968872, 0], [1.3731423616409302, -1.5452397505389612, -1.783734146748678, -1.373345200215475, 1.57102632522583, 0], [1.3730944395065308, -1.5509522596942347, -1.7939680258380335, -1.357504669819967, 1.5710142850875854, 0], [1.373058557510376, -1.5573113600360315, -1.8049691359149378, -1.3398311773883265, 1.5707746744155884, 0], [1.372998595237732, -1.5654791037188929, -1.819305721913473, -1.3174126783954065, 1.5707746744155884, 0], [1.3729146718978882, -1.5733111540423792, -1.8324902693377894, -1.2965057531939905, 1.5704506635665894, 0], [1.372842788696289, -1.583048168812887, -1.8491066137896937, -1.269799534474508, 1.5705469846725464, 0], [1.37277090549469, -1.5934789816485804, -1.867389980946676, -1.2412131468402308, 1.5703188180923462, 0], [1.37265145778656, -1.6028083006488245, -1.883310619984762, -1.2158620993243616, 1.5703308582305908, 0], [1.3725556135177612, -1.6121380964862269, -1.8992307821856897, -1.1907036940204065, 1.5701031684875488, 0], [1.3724716901779175, -1.6227005163775843, -1.917837921773092, -1.1613882223712366, 1.5700792074203491, 0], [1.3723877668380737, -1.632054630910055, -1.9336383978473108, -1.136327091847555, 1.5698634386062622, 0], [1.3723278045654297, -1.6414082686053675, -1.9496672789203089, -1.11125356355776, 1.5698155164718628, 0], [1.3721840381622314, -1.651959244404928, -1.9680827299701136, -1.081893269215719, 1.5695518255233765, 0], [1.3720881938934326, -1.661264721547262, -1.9841349760638636, -1.056868855153219, 1.5696357488632202, 0], [1.3720521926879883, -1.6706789175616663, -1.9998868147479456, -1.0316298643695276, 1.5694197416305542, 0], [1.371884822845459, -1.6806071440326136, -2.0175583998309534, -1.0039961973773401, 1.5693118572235107, 0], [1.3718369007110596, -1.688631836568014, -2.0313313643084925, -0.9820402304278772, 1.5691440105438232, 0], [1.3717529773712158, -1.6961529890643519, -2.044431988392965, -0.9615939299212855, 1.5688804388046265, 0], [1.3716810941696167, -1.7042015234576624, -2.0580132643329065, -0.9398306051837366, 1.5689523220062256, 0], [1.3716450929641724, -1.7102378050433558, -2.068702522908346, -0.9229782263385218, 1.5687965154647827, 0], [1.3716092109680176, -1.7158435026751917, -2.0785163084613245, -0.9076350370990198, 1.5688084363937378, 0], [1.3715612888336182, -1.7216280142413538, -2.08836537996401, -0.8919575850116175, 1.5686043500900269, 0], [1.3714298009872437, -1.7260239760028284, -2.095755402241842, -0.8802197615252894, 1.5685803890228271, 0], [1.3714653253555298, -1.7297967115985315, -2.102306667958395, -0.8700874487506312, 1.5685803890228271, 0], [1.3714298009872437, -1.733389679585592, -2.108652416859762, -0.8602545897113245, 1.5684605836868286, 0], [1.371417760848999, -1.7358091513263147, -2.11280328432192, -0.8535469214068812, 1.5685324668884277, 0], [1.371417760848999, -1.7376535574542444, -2.116018597279684, -0.8484690825091761, 1.5684245824813843, 0], [1.3713698387145996, -1.7390549818622034, -2.1184542814837855, -0.8446486631976526, 1.5684845447540283, 0], [1.3713343143463135, -1.739617649708883, -2.1194022337542933, -0.8431037108050745, 1.5683887004852295, 0], [1.3713698387145996, -1.7395218054400843, -2.119462315236227, -0.843055550252096, 1.5683526992797852, 0], [1.3713698387145996, -1.7395218054400843, -2.119486157094137, -0.8431156317340296, 1.5683647394180298, 0], [1.3713462352752686, -1.7395337263690394, -2.119462315236227, -0.8430793921100062, 1.5683887004852295, 0], [1.3713582754135132, -1.7395337263690394, -2.1194499174701136, -0.8430793921100062, 1.5683526992797852, 0], [1.3713222742080688, -1.7395098845111292, -2.1194499174701136, -0.8431037108050745, 1.5683647394180298, 0], [1.3713343143463135, -1.7395575682269495, -2.1194499174701136, -0.8431037108050745, 1.56834077835083, 0]], [[1.3714537620544434, -1.7200830618487757, -2.0858100096331995, -0.8964007536517542, 1.5686163902282715, 0], [1.3714892864227295, -1.7150166670428675, -2.0769084135638636, -0.9103663603412073, 1.5687006711959839, 0], [1.3715612888336182, -1.7083094755755823, -2.0652712027179163, -0.9286797682391565, 1.5687605142593384, 0], [1.3716810941696167, -1.7019503752337855, -2.054389301930563, -0.9462264219867151, 1.5687605142593384, 0], [1.3717290163040161, -1.694859806691305, -2.0422490278827112, -0.9651392141925257, 1.5690242052078247, 0], [1.3717650175094604, -1.6873739401446741, -2.0291836897479456, -0.9858611265765589, 1.5691081285476685, 0], [1.3718608617782593, -1.6777690092669886, -2.012735668812887, -1.0120213667498987, 1.5694197416305542, 0], [1.371944785118103, -1.6688817183123987, -1.9974396864520472, -1.035990063344137, 1.5693837404251099, 0], [1.3720881938934326, -1.6595047155963343, -1.981567684804098, -1.0612414518939417, 1.5695035457611084, 0], [1.372136116027832, -1.6488216559039515, -1.962923828755514, -1.090517822896139, 1.5695759057998657, 0], [1.3722079992294312, -1.6397069136248987, -1.946979824696676, -1.1157696882831019, 1.5697436332702637, 0], [1.3722679615020752, -1.6303537527667444, -1.931155029927389, -1.1409266630755823, 1.5699113607406616, 0], [1.372423768043518, -1.619730297719137, -1.9124634901629847, -1.1704209486590784, 1.5700312852859497, 0], [1.3724716901779175, -1.6104849020587366, -1.8967354933368128, -1.1954472700702112, 1.5702589750289917, 0], [1.3725556135177612, -1.601095978413717, -1.8806708494769495, -1.2205103079425257, 1.570246934890747, 0], [1.3726633787155151, -1.5918744246112269, -1.8648589293109339, -1.2457059065448206, 1.5702948570251465, 0], [1.3727588653564453, -1.5815027395831507, -1.8467310110675257, -1.274088207875387, 1.5705349445343018, 0], [1.372830867767334, -1.573179070149557, -1.8326223532306116, -1.296901051198141, 1.5705349445343018, 0], [1.3728787899017334, -1.5653355757342737, -1.8192337195025843, -1.3177130858050745, 1.5706547498703003, 0], [1.3729506731033325, -1.5571554342852991, -1.8049328962909144, -1.3401549498187464, 1.5706907510757446, 0], [1.3730106353759766, -1.5507848898517054, -1.7938836256610315, -1.357828442250387, 1.570894479751587, 0], [1.3731184005737305, -1.5448921362506312, -1.7837942282306116, -1.3735368887530726, 1.5711342096328735, 0], [1.3731544017791748, -1.5387609640704554, -1.7733929792987269, -1.3901198546039026, 1.571194052696228, 0], [1.3731664419174194, -1.5342100302325647, -1.7655108610736292, -1.402546230946676, 1.5712419748306274, 0], [1.3731783628463745, -1.5302699247943323, -1.7586243788348597, -1.4133666197406214, 1.5711700916290283, 0], [1.3732502460479736, -1.526916805897848, -1.7527220884906214, -1.422605339680807, 1.5713379383087158, 0], [1.373286247253418, -1.5237553755389612, -1.7472875753985804, -1.4311726729022425, 1.5712900161743164, 0], [1.373358130455017, -1.5216591993915003, -1.7436884085284632, -1.4368770758258265, 1.5714941024780273, 0], [1.3733341693878174, -1.5201266447650355, -1.7410491148578089, -1.4410350958453577, 1.5714341402053833, 0], [1.3733701705932617, -1.5191686789142054, -1.7392852942096155, -1.4438031355487269, 1.5714820623397827, 0], [1.373358130455017, -1.5190485159503382, -1.7389615217791956, -1.4443305174456995, 1.571349859237671, 0], [1.3724836111068726, -1.5190966765033167, -1.7387455145465296, -1.4444263617144983, 1.571518063545227, 0], [1.3694418668746948, -1.519348446522848, -1.7377856413470667, -1.4450257460223597, 1.5713618993759155, 0], [1.3649388551712036, -1.519707504902975, -1.7365978399859827, -1.4457686583148401, 1.5715299844741821, 0], [1.358891248703003, -1.520306412373678, -1.7349308172809046, -1.446787182484762, 1.5714101791381836, 0], [1.351130723953247, -1.521132771168844, -1.732699219380514, -1.4481290022479456, 1.571518063545227, 0], [1.3401011228561401, -1.522330109273092, -1.7297237555133265, -1.4500582853900355, 1.571578025817871, 0], [1.3289992809295654, -1.5235517660724085, -1.7265690008746546, -1.4520838896380823, 1.5717097520828247, 0], [1.3162087202072144, -1.5249045530902308, -1.7230537573443812, -1.4542048613177698, 1.571901559829712, 0], [1.2992626428604126, -1.5267012755023401, -1.7183030287372034, -1.457092587147848, 1.5719255208969116, 0], [1.283046841621399, -1.5284255186664026, -1.713815991078512, -1.4599688688861292, 1.572201132774353, 0], [1.2650705575942993, -1.5303056875811976, -1.7089455763446253, -1.4630244413958948, 1.57229745388031, 0], [1.2455973625183105, -1.532341782246725, -1.7035229841815394, -1.4664519468890589, 1.5725011825561523, 0], [1.2209378480911255, -1.5349763075457972, -1.69666034380068, -1.470705811177389, 1.5727168321609497, 0], [1.1979068517684937, -1.5373838583575647, -1.690218750630514, -1.474720303212301, 1.5729326009750366, 0], [1.1733309030532837, -1.5400298277484339, -1.683500115071432, -1.479058567677633, 1.5732325315475464, 0], [1.1438804864883423, -1.5431202093707483, -1.6753304640399378, -1.484103027974264, 1.573412299156189, 0], [1.1187409162521362, -1.5457666555987757, -1.6683123747455042, -1.4883931318866175, 1.5737358331680298, 0], [1.0935170650482178, -1.5483415762530726, -1.6612585226642054, -1.492791477833883, 1.5738916397094727, 0], [1.0640654563903809, -1.5514310042010706, -1.6531251112567347, -1.4978607336627405, 1.5741796493530273, 0], [1.0389372110366821, -1.554066006337301, -1.6462262312518519, -1.5022710005389612, 1.5743833780288696, 0], [1.0138201713562012, -1.556664768849508, -1.6392324606524866, -1.5066326300250452, 1.5745152235031128, 0], [0.9885712265968323, -1.5593231360064905, -1.6322386900531214, -1.510970417653219, 1.5748627185821533, 0], [0.9593698382377625, -1.5624011198626917, -1.6240571180926722, -1.516076389943258, 1.5750548839569092, 0], [0.9342280030250549, -1.5650599638568323, -1.6171944777118128, -1.520426098500387, 1.575438380241394, 0], [0.9091218113899231, -1.5677779356585901, -1.6101892630206507, -1.5247882048236292, 1.5757139921188354, 0], [0.879763126373291, -1.5708683172809046, -1.6020434538470667, -1.5298455397235315, 1.5758458375930786, 0], [0.8546561598777771, -1.5735510031329554, -1.5950735251056116, -1.534208122883932, 1.5761821269989014, 0], [0.8296806216239929, -1.5761855284320276, -1.5881393591510218, -1.5386536757098597, 1.5762779712677002, 0], [0.8003444075584412, -1.5792158285724085, -1.580017391835348, -1.5437114874469202, 1.5766974687576294, 0], [0.775307834148407, -1.5818379561053675, -1.573024098073141, -1.548037354146139, 1.5768057107925415, 0], [0.7501158714294434, -1.5844128767596644, -1.5659220854388636, -1.5523284117328089, 1.576985478401184, 0], [0.7250789403915405, -1.5870960394488733, -1.559000317250387, -1.556690518056051, 1.577368974685669, 0], [0.6957647800445557, -1.5902331511126917, -1.5509145895587366, -1.5618317762957972, 1.5775487422943115, 0], [0.6707027554512024, -1.592856232319967, -1.5439093748675745, -1.5661824385272425, 1.5779086351394653, 0], [0.6455089449882507, -1.5955269972430628, -1.5369275251971644, -1.5705683867083948, 1.5780165195465088, 0], [0.6161449551582336, -1.5985925833331507, -1.5287459532367151, -1.5756142775165003, 1.5784239768981934, 0], [0.5910578966140747, -1.60132342973818, -1.5218127409564417, -1.580024544392721, 1.5785198211669922, 0], [0.5658625364303589, -1.6039579550372522, -1.5148308912860315, -1.5844109694110315, 1.5787838697433472, 0], [0.5365933179855347, -1.6070597807513636, -1.506733242665426, -1.5894563833819788, 1.5789636373519897, 0], [0.5114088654518127, -1.6096470991717737, -1.4997161070453089, -1.5938070456134241, 1.5792632102966309, 0], [0.4863075017929077, -1.6122577826129358, -1.4927704969989222, -1.5981572310077112, 1.5795512199401855, 0], [0.4611941874027252, -1.614892307912008, -1.4857528845416468, -1.6026161352740687, 1.5796470642089844, 0], [0.43165868520736694, -1.6179583708392542, -1.4776433149920862, -1.6076496283160608, 1.579886794090271, 0], [0.4065326452255249, -1.6205456892596644, -1.4706137816058558, -1.6119282881366175, 1.580186367034912, 0], [0.38135790824890137, -1.623227898274557, -1.463740650807516, -1.6163032690631312, 1.5804264545440674, 0], [0.35191676020622253, -1.6263898054706019, -1.4555233160602015, -1.621432129536764, 1.5807380676269531, 0], [0.3268607556819916, -1.6290367285357874, -1.44861346880068, -1.6257832686053675, 1.5808218717575073, 0], [0.30161261558532715, -1.6316950956927698, -1.4416082541095179, -1.6301339308368128, 1.581193447113037, 0], [0.2765798270702362, -1.6343300978290003, -1.4345429579364222, -1.6345799604998987, 1.581397533416748, 0], [0.2472323179244995, -1.6375277678119105, -1.4265182654010218, -1.639602009450094, 1.5817570686340332, 0], [0.22222235798835754, -1.6401146093951624, -1.4194768110858362, -1.6439889113055628, 1.5818289518356323, 0], [0.19700829684734344, -1.6427014509784144, -1.412459675465719, -1.6483033339129847, 1.582176923751831, 0], [0.16774283349514008, -1.6456835905658167, -1.4043148199664515, -1.6533730665790003, 1.5822967290878296, 0], [0.14300701022148132, -1.6483305136310022, -1.3974412123309534, -1.6577118078814905, 1.5826562643051147, 0], [0.11763565987348557, -1.651001278554098, -1.3903759161578577, -1.6621349493609827, 1.5827521085739136, 0], [0.08829639852046967, -1.6541269461261194, -1.38220721880068, -1.6673243681537073, 1.5830397605895996, 0], [0.06317558884620667, -1.6567500273333948, -1.3751781622516077, -1.6716635862933558, 1.5833756923675537, 0], [0.03810197114944458, -1.6593850294696253, -1.3681967894183558, -1.6759064833270472, 1.583579421043396, 0], [0.012944460846483707, -1.6620915571795862, -1.3612998167621058, -1.6802452246295374, 1.5837831497192383, 0], [-0.01634866396059209, -1.665229622517721, -1.3531306425677698, -1.6853750387774866, 1.5839753150939941, 0], [-0.041471306477681935, -1.6678999106036585, -1.3461020628558558, -1.6897261778460901, 1.5843348503112793, 0], [-0.06667834917177373, -1.6704629103290003, -1.3391574064837855, -1.6941006819354456, 1.5844905376434326, 0], [-0.09594947496523076, -1.673577133809225, -1.3310607115374964, -1.6991356054889124, 1.5848984718322754, 0], [-0.12104970613588506, -1.676199738179342, -1.324139420186178, -1.7035577932940882, 1.5851141214370728, 0], [-0.1461861769305628, -1.6787751356707972, -1.3170502821551722, -1.7079089323626917, 1.5853538513183594, 0], [-0.17556697527040654, -1.6818650404559534, -1.308929745350973, -1.7129915396319788, 1.5856175422668457, 0], [-0.20071679750551397, -1.6844881216632288, -1.3019970099078577, -1.7173660437213343, 1.585905909538269, 0], [-0.22587901750673467, -1.68724233308901, -1.2949450651751917, -1.7217052618609827, 1.5861455202102661, 0], [-0.2509573141681116, -1.6898892561541956, -1.287976090108053, -1.7262004057513636, 1.5862174034118652, 0], [-0.28036433855165654, -1.6929553190814417, -1.2798073927508753, -1.7312348524676722, 1.5865050554275513, 0], [-0.3053472677813929, -1.6955903212176722, -1.2728989760028284, -1.7355979124652308, 1.586864948272705, 0], [-0.3306077162372034, -1.698273007069723, -1.2659538427936, -1.7398770491229456, 1.5869249105453491, 0], [-0.36012393632997686, -1.701458756123678, -1.2577975432025355, -1.745007340108053, 1.58724844455719, 0], [-0.38519269624818975, -1.7039859930621546, -1.2507932821856897, -1.7493704001056116, 1.5873563289642334, 0], [-0.4104417006122034, -1.706597153340475, -1.2438605467425745, -1.7536733786212366, 1.5877881050109863, 0], [-0.43979150453676397, -1.709555451069967, -1.2357166449176233, -1.7588518301593226, 1.588027834892273, 0], [-0.464945141469137, -1.7122743765460413, -1.2288201490985315, -1.7631552855121058, 1.5882076025009155, 0], [-0.4900158087359827, -1.7149446646319788, -1.221708122883932, -1.7675302664386194, 1.588531494140625, 0], [-0.5151947180377405, -1.7176278273211878, -1.2147391478167933, -1.7718337217914026, 1.588747262954712, 0], [-0.5444744269000452, -1.7206700483905237, -1.2067392508136194, -1.776952091847555, 1.589046835899353, 0], [-0.5695694128619593, -1.7233646551715296, -1.1998432318316858, -1.7812793890582483, 1.5891307592391968, 0], [-0.5946539084063929, -1.7260239760028284, -1.1927431265460413, -1.7856906096087855, 1.5894906520843506, 0], [-0.6237428824054163, -1.72914964357485, -1.1845267454730433, -1.7908929030047815, 1.5896464586257935, 0], [-0.6488874594317835, -1.7317727247821253, -1.177546803151266, -1.7951963583575647, 1.589993953704834, 0], [-0.6740325132953089, -1.7343714872943323, -1.1706388632403772, -1.799511734639303, 1.5901257991790771, 0], [-0.6991780439959925, -1.7370184103595179, -1.1636346022235315, -1.803779427205221, 1.5904977321624756, 0], [-0.7285679022418421, -1.7400849501239222, -1.1555030981646937, -1.8088739554034632, 1.5907254219055176, 0], [-0.7536419073687952, -1.7426837126361292, -1.148498837147848, -1.8133094946490687, 1.5909889936447144, 0], [-0.7787163893329065, -1.7453306357013147, -1.1415789763080042, -1.8176363150226038, 1.5910968780517578, 0], [-0.8080475966082972, -1.7484329382525843, -1.1334713141070765, -1.8227198759662073, 1.591492772102356, 0], [-0.8331826368915003, -1.751115624104635, -1.1264193693744105, -1.827094856892721, 1.5916125774383545, 0], [-0.8583658377276819, -1.753786865864889, -1.119439427052633, -1.8315189520465296, 1.5919841527938843, 0], [-0.88757831255068, -1.756852928792135, -1.1113203207599085, -1.8365772406207483, 1.5921403169631958, 0], [-0.9127019087420862, -1.7595117727862757, -1.1043885389911097, -1.8408568541156214, 1.592511773109436, 0], [-0.9379218260394495, -1.7621586958514612, -1.0973723570453089, -1.845196549092428, 1.5926436185836792, 0], [-0.9630335013019007, -1.7648299376117151, -1.0903447310077112, -1.8496559301959437, 1.5928837060928345, 0], [-0.9923189322101038, -1.7678840796100062, -1.0822256247149866, -1.854774300252096, 1.5930753946304321, 0], [-1.0175989309894007, -1.7704232374774378, -1.075317684804098, -1.8590782324420374, 1.5934349298477173, 0], [-1.0428078810321253, -1.7729981581317347, -1.0682185331927698, -1.8634422461139124, 1.5935428142547607, 0], [-1.0705588499652308, -1.775980297719137, -1.0604594389544886, -1.8682368437396448, 1.5939745903015137, 0], [-1.0924938360797327, -1.7783640066729944, -1.0543912092791956, -1.872049633656637, 1.5940704345703125, 0], [-1.112845245991842, -1.7804964224444788, -1.0487788359271448, -1.8755863348590296, 1.5943580865859985, 0], [-1.1346123854266565, -1.7827718893634241, -1.042722527180807, -1.8793142477618616, 1.5943820476531982, 0], [-1.1514628569232386, -1.7845566908465784, -1.038081471120016, -1.8822997252093714, 1.5946341753005981, 0], [-1.166717831288473, -1.7861974875079554, -1.0337522665606897, -1.884876553212301, 1.5949101448059082, 0], [-1.180293385182516, -1.787670914326803, -1.0299032370196741, -1.8873465696917933, 1.594994068145752, 0], [-1.194169823323385, -1.789192024861471, -1.0261614958392542, -1.889684025441305, 1.595233678817749, 0], [-1.2043278853045862, -1.790257755910055, -1.0232952276812952, -1.8914101759540003, 1.595221757888794, 0], [-1.2129386107074183, -1.7911203543292444, -1.0209205786334437, -1.8929451147662562, 1.5953055620193481, 0], [-1.220926586781637, -1.791898552571432, -1.018726650868551, -1.8943355719195765, 1.595389485359192, 0], [-1.2258551756488245, -1.7924378553973597, -1.0173595587359827, -1.895078484212057, 1.5953415632247925, 0], [-1.229309384022848, -1.792701546345846, -1.0164240042315882, -1.8955701033221644, 1.5954853296279907, 0], [-1.2312162558185022, -1.7928808371173304, -1.0159805456744593, -1.8959062735186976, 1.5955456495285034, 0], [-1.2314799467669886, -1.7928689161883753, -1.0161002318011683, -1.8959301153766077, 1.5955332517623901, 0], [-1.2314561049090784, -1.7931087652789515, -1.0174434820758265, -1.8944552580462855, 1.5955092906951904, 0], [-1.2314441839801233, -1.7933839003192347, -1.0202849547015589, -1.8912666479693812, 1.5955092906951904, 0], [-1.2314441839801233, -1.7938750425921839, -1.0256698767291468, -1.885308090840475, 1.595521330833435, 0], [-1.2314441839801233, -1.7943900267230433, -1.0319893995868128, -1.8785107771502894, 1.595521330833435, 0], [-1.2314680258380335, -1.7949288527118128, -1.0397008101092737, -1.8702510038958948, 1.595461368560791, 0], [-1.2314680258380335, -1.7958276907550257, -1.050445858632223, -1.8585031668292444, 1.5954374074935913, 0], [-1.231516186391012, -1.7967374960528772, -1.06140643755068, -1.8465750853167933, 1.5953415632247925, 0], [-1.2315757910357874, -1.7977197805987757, -1.0739625136004847, -1.833172623311178, 1.5954973697662354, 0], [-1.2316001097308558, -1.7987621466266077, -1.0876219908343714, -1.8180087248431605, 1.595449447631836, 0], [-1.2317317167865198, -1.8002470175372522, -1.105875317250387, -1.7986372152911585, 1.595389485359192, 0], [-1.2317679564105433, -1.8015530745135706, -1.1228931585894983, -1.7801044623004358, 1.5954134464263916, 0], [-1.231851879750387, -1.8030617872821253, -1.1416266600238245, -1.7600744406329554, 1.5953176021575928, 0], [-1.2319358030902308, -1.8049305121051233, -1.1652534643756312, -1.7346509138690394, 1.5953055620193481, 0], [-1.2319839636432093, -1.8065832296954554, -1.1870458761798304, -1.7109292189227503, 1.595293641090393, 0], [-1.232067886983053, -1.8085716406451624, -1.2101934591876429, -1.6859505812274378, 1.595449447631836, 0], [-1.2322114149676722, -1.8107636610614222, -1.2374799887286585, -1.6565974394427698, 1.5952576398849487, 0], [-1.232295338307516, -1.8125723044024866, -1.260796372090475, -1.6313203016864222, 1.5954374074935913, 0], [-1.2323434988604944, -1.8144405523883265, -1.283945385609762, -1.6063554922686976, 1.595221757888794, 0], [-1.2324517408954065, -1.8163211981402796, -1.3072150389300745, -1.5812109152423304, 1.595221757888794, 0], [-1.2325595060931605, -1.818500820790426, -1.334419075642721, -1.5518844763385218, 1.5951257944107056, 0], [-1.2326315085040491, -1.8203218618976038, -1.3576772848712366, -1.5266817251788538, 1.5952576398849487, 0], [-1.2326911131488245, -1.8222621122943323, -1.3810675779925745, -1.5015156904803675, 1.59516179561615, 0], [-1.2327869574176233, -1.8243945280658167, -1.4080570379840296, -1.4722159544574183, 1.5952696800231934, 0], [-1.2328951994525355, -1.8263705412494105, -1.4314120451556605, -1.447158161793844, 1.5950419902801514, 0], [-1.2329910437213343, -1.8281315008746546, -1.4547675291644495, -1.4221742788897913, 1.59516179561615, 0], [-1.2330992857562464, -1.830179516469137, -1.4818175474749964, -1.3927448431598108, 1.5950778722763062, 0], [-1.2331350485431116, -1.831951920186178, -1.5051620642291468, -1.3675816694842737, 1.5951738357543945, 0], [-1.2332547346698206, -1.8338807264911097, -1.5283501783954065, -1.3423236052142542, 1.595065951347351, 0], [-1.2333386580096644, -1.8357251326190394, -1.5516584555255335, -1.3173297087298792, 1.5951257944107056, 0], [-1.2334469000445765, -1.8379409948932093, -1.5789502302752894, -1.2879388968097132, 1.5950778722763062, 0], [-1.23350698152651, -1.8397019545184534, -1.6021149794207972, -1.262766186391012, 1.59516179561615, 0], [-1.2335670630084437, -1.8416546026812952, -1.6248129049884241, -1.2384451071368616, 1.5950419902801514, 0], [-1.2336505095111292, -1.8436186949359339, -1.6492975393878382, -1.2117889563189905, 1.5950899124145508, 0], [-1.2337706724749964, -1.8450802008258265, -1.668816391621725, -1.1907990614520472, 1.5949220657348633, 0], [-1.233842674885885, -1.8465412298785608, -1.686547581349508, -1.1715949217425745, 1.5949820280075073, 0], [-1.2338302771197718, -1.8478706518756312, -1.7031028906451624, -1.1538050810443323, 1.595005989074707, 0], [-1.2339027563678187, -1.8492720762835901, -1.7203062216388147, -1.1352599302874964, 1.5951018333435059, 0], [-1.2339747587787073, -1.8503263632403772, -1.7335990110980433, -1.1210768858539026, 1.5950180292129517, 0], [-1.234046761189596, -1.8512128035174769, -1.745199982319967, -1.1084025541888636, 1.5948861837387085, 0], [-1.2340825239764612, -1.8521950880633753, -1.7570889631854456, -1.0955608526812952, 1.5948981046676636, 0], [-1.2340706030475062, -1.8528779188739222, -1.765594784413473, -1.0865524450885218, 1.5949101448059082, 0], [-1.2341306845294397, -1.853429142628805, -1.7724688688861292, -1.0790656248675745, 1.5949580669403076, 0], [-1.2341426054583948, -1.8538358847247522, -1.7787554899798792, -1.0722973982440394, 1.5948622226715088, 0], [-1.23415452638735, -1.8540871779071253, -1.7825349012957972, -1.0681889692889612, 1.5950419902801514, 0], [-1.2341426054583948, -1.8541830221759241, -1.7848499456988733, -1.0657570997821253, 1.5948622226715088, 0], [-1.234166447316305, -1.8542669455157679, -1.785689655934469, -1.0649426619159144, 1.5948981046676636, 0], [-1.23417836824526, -1.854278866444723, -1.7855337301837366, -1.0651219526873987, 1.5949701070785522, 0], [-1.23417836824526, -1.854278866444723, -1.7855694929706019, -1.0650981108294886, 1.594946026802063, 0], [-1.23417836824526, -1.8542912642108362, -1.7855218092547815, -1.0651219526873987, 1.5949101448059082, 0], [-1.2342141310321253, -1.854387108479635, -1.7855818907367151, -1.0650861899005335, 1.594934105873108, 0], [-1.2342265287982386, -1.8544228712665003, -1.7855456511126917, -1.0650861899005335, 1.5949580669403076, 0], [-1.2341902891742151, -1.8543990294085901, -1.785593334828512, -1.0650981108294886, 1.594946026802063, 0], [-1.2341902891742151, -1.8543990294085901, -1.7855575720416468, -1.0650981108294886, 1.594934105873108, 0], [-1.2341902891742151, -1.8544228712665003, -1.7855694929706019, -1.0650861899005335, 1.5949580669403076, 0], [-1.2341902891742151, -1.8544467131244105, -1.7855818907367151, -1.0650981108294886, 1.5949220657348633, 0], [-1.2342022101031702, -1.8544347921954554, -1.7855575720416468, -1.0651105085956019, 1.5949580669403076, 0], [-1.2342141310321253, -1.8544467131244105, -1.785593334828512, -1.0651105085956019, 1.594934105873108, 0], [-1.23417836824526, -1.8544467131244105, -1.7855694929706019, -1.0651105085956019, 1.5949580669403076, 0], [-1.2342141310321253, -1.8544228712665003, -1.7855818907367151, -1.0651105085956019, 1.5949101448059082, 0], [-1.2342141310321253, -1.8544109503375452, -1.7855694929706019, -1.0650742689715784, 1.594934105873108, 0], [-1.234166447316305, -1.8544586340533655, -1.7855456511126917, -1.0650861899005335, 1.594934105873108, 0], [-1.2341902891742151, -1.8544347921954554, -1.7855818907367151, -1.0651105085956019, 1.5949101448059082, 0], [-1.2342265287982386, -1.8544467131244105, -1.7855818907367151, -1.0650861899005335, 1.594946026802063, 0], [-1.2342141310321253, -1.8544705549823206, -1.7855818907367151, -1.0650981108294886, 1.594946026802063, 0], [-1.2342141310321253, -1.8544705549823206, -1.7855818907367151, -1.0650861899005335, 1.5949220657348633, 0], [-1.2342141310321253, -1.8544347921954554, -1.7855218092547815, -1.0650861899005335, 1.5949101448059082, 0], [-1.2342022101031702, -1.8544586340533655, -1.7855694929706019, -1.0650981108294886, 1.5949220657348633, 0], [-1.2342141310321253, -1.8544705549823206, -1.7855575720416468, -1.0651105085956019, 1.5949101448059082, 0]]]