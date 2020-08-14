import random
from classes import *
import time
import numpy as np
from math import *
import sys
import rclpy
from rclpy.node import Node

joint_limits = [[[0, 2.36], [-1.57, 0.52], [-1.57, 1.57], [-2.618, 0.0], [-1.13, 0.44], [-1.57, 0.52]], [[0, 2.36], [-1.57, .52], [-1.57, 1.57], [-2.618, 0.0], [-1.13, 0.44], [-1.57, 0.52]], [[-0.52, 1.57]], [[-0.52, 1.57]], [[-0.52, 1.57], [-0.79, 0.79], [-1.57, 1.57]]]  # added soft limits in utils.py

duration_min = .2
duration_max = .8


hip_z_min_temp = -.78
hip_z_max_temp = .78
hip_x_min_temp = 0.0
hip_x_max_temp = .8 #.8, 
hip_y_min_temp = -.22
hip_y_max_temp = .22
knee_min_temp = -.8 #-.8
knee_max_temp = 0.0
ankle_y_min_temp = -.22
ankle_y_max_temp = .22
ankle_x_min_temp =  -.2
ankle_x_max_temp = .8 #.2




class ActionSub(Node):
    def __init__(self):
        super().__init__('action_node')
        self.action_sub = self.create_subscription(Action_msg, '/action', self.command_action_cb)
        self.receive_state_sub = self.create_subscription(Action_msg, '/tori_state', self.receive_state_cb)

