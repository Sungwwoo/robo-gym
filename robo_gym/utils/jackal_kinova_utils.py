#!/usr/bin/env python3

import numpy as np
import yaml
from robo_gym.utils import utils


class Jackal_Kinova:
    """Mobile manipulator Jackal_Kinova utilities.

    Attributes:
        max_lin_vel (float): Maximum robot's linear velocity (m/s).
        min_lin_vel (float): Minimum robot's linear velocity (m/s).
        max_ang_vel (float): Maximum robot's angular velocity (rad/s).
        min_ang_vel (float): Minimum robot's angular velocity (rad/s).

    """

    def __init__(self):
        # Using same parametes with APF-RL
        with open('/home/sungwoo/catkin_ws/src/robo-gym-robot-servers/apf_robot_server/config/jackal_params.yaml') as file:
            print("Loading params for APF utility class")
            params = yaml.load(file, Loader=yaml.FullLoader)
        
        self.max_lin_vel = params['vel_x_max']
        self.min_lin_vel = params['vel_x_min']

        self.max_ang_vel = params['vel_theta_max']
        self.min_ang_vel = -params['vel_theta_max']

        self.max_lin_acc = params['acc_x_lim']
        self.max_ang_acc = params['acc_theta_lim']


    def get_max_lin_vel(self):
        return self.max_lin_vel

    def get_min_lin_vel(self):
        return self.min_lin_vel

    def get_max_ang_vel(self):
        return self.max_ang_vel

    def get_min_ang_vel(self):
        return self.min_ang_vel
    
    def get_max_lin_acc(self):
        return self.max_lin_acc

    def get_max_ang_acc(self):
        return self.max_ang_acc


    def get_corners_positions(self, x, y, yaw):
        """Get robot's corners coordinates given the coordinates of its center.

        Args:
            x (float): x coordinate of robot's geometric center.
            y (float): y coordinate of robot's geometric center.
            yaw (float): yaw angle of robot's geometric center.

        The coordinates are given with respect to the map origin and cartesian system.

        Returns:
            list[list]: x and y coordinates of the 4 robot's corners.

        """

        robot_x_dimension = 0.5
        robot_y_dimension = 0.325
        dx = robot_x_dimension / 2
        dy = robot_y_dimension / 2

        delta_corners = [[dx, -dy], [dx, dy], [-dx, dy], [-dx, -dy]]
        corners = []

        for corner_xy in delta_corners:
            # Rotate point around origin
            r_xy = utils.rotate_point(corner_xy[0], corner_xy[1], yaw)
            # Translate back from origin to corner
            corners.append([sum(x) for x in zip(r_xy, [x, y])])

        return corners
