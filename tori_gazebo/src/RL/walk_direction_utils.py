import random
from classes import *
import time
import numpy as np
from math import *
import sys

joint_limits = [[[0, 2.36], [-1.57, 0.52], [-1.57, 1.57], [-2.618, 0.0], [-1.13, 0.44], [-1.57, 0.52]], [[0, 2.36], [-1.57, .52], [-1.57, 1.57], [-2.618, 0.0], [-1.13, 0.44], [-1.57, 0.52]], [[-0.52, 1.57]], [[-0.52, 1.57]], [[-0.52, 1.57], [-0.79, 0.79], [-1.57, 1.57]]]  # added soft limits in utils.py

duration_min = .2
duration_max = .8


hip_z_min_temp = -.4 #-.78
hip_z_max_temp = .4 #.78
hip_x_min_temp = 0.0
hip_x_max_temp = .4 #.8, 
hip_y_min_temp = -.22
hip_y_max_temp = .22
knee_min_temp = -.4 #-.8
knee_max_temp = 0.0
ankle_y_min_temp = -.22
ankle_y_max_temp = .22
ankle_x_min_temp =  -.2
ankle_x_max_temp = .4 #.2





def new_state_after_action(s, action, target_yaw_decomposed, server, i, last_saved_index):
    # The starting position should be standing up straight. This should be achieved with all zeros.
    # To achieve more realistic motions, set soft joint limits.    

    # map -1 - 1 to joint_min - joint_max
    # 0 - 2 to joint_min - joint_max
    get_min_x = 1 if target_yaw_decomposed[0] >= 0 else 0 # determine whether to get min or max (doesn't matter anymore since we're using foot average)
    get_min_y = 1 if target_yaw_decomposed[1] >= 0 else 0
    print('decomposed:{}'.format(target_yaw_decomposed))
    print('get_min_x:{} get_min_y:{}'.format(get_min_x, get_min_y))

        
    
    adj_actions = [(x + 1)/2. for x in action] # bound actions between 0 and 1

    hip_l_x = (hip_x_max_temp - hip_x_min_temp) * adj_actions[0] + hip_x_min_temp
    hip_l_y = (hip_y_max_temp - hip_y_min_temp) * adj_actions[1] + hip_y_min_temp
    hip_l_z = (hip_z_max_temp - hip_z_min_temp) * adj_actions[10] + hip_z_min_temp
    knee_l = (knee_max_temp - knee_min_temp) * adj_actions[2] + knee_min_temp
    ankle_l_y = (ankle_y_max_temp - ankle_y_min_temp) * adj_actions[3] + ankle_y_min_temp
    ankle_l_x = (ankle_x_max_temp - ankle_x_min_temp) * adj_actions[4] + ankle_x_min_temp
    hip_r_x = (hip_x_max_temp - hip_x_min_temp) * adj_actions[5] + hip_x_min_temp
    hip_r_y = (hip_y_max_temp - hip_y_min_temp) * adj_actions[6] + hip_y_min_temp
    hip_r_z = (hip_z_max_temp - hip_z_min_temp) * adj_actions[11] + hip_z_min_temp
    knee_r = (knee_max_temp - knee_min_temp) * adj_actions[7] + knee_min_temp
    ankle_r_y = (ankle_y_max_temp - ankle_y_min_temp) * adj_actions[8] + ankle_y_min_temp
    ankle_r_x = (ankle_x_max_temp - ankle_x_min_temp) * adj_actions[9] + ankle_x_min_temp
    
    ball_l = 0 #(ball_l_max - ball_l_min) * adj_actions[10] + ball_l_min
    ball_r = 0 #(ball_r_max - ball_r_min) * adj_actions[11] + ball_r_min
    
    spine_x = 0
    spine_y = 0
    spine_z = 0
    duration = .1
    print('duration:{}'.format(duration))
    
    if i > 0 and i != last_saved_index:
        server.send_message('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(hip_l_x, hip_l_y, hip_l_z, knee_l, ankle_l_y, ankle_l_x,
                            hip_r_x, hip_r_y, hip_r_z, knee_r, ankle_r_y, ankle_r_x, 
                            ball_l, 
                            ball_r, 
                            spine_x, spine_y, spine_z,
                            duration,
                            get_min_x, get_min_y))
			


    
    msg = server.receive_message()

    server.send_message('buffer1')
    orientation = server.receive_message().split(':')

    #updated
    fallen_status = int(orientation[0])
    roll = float(orientation[1])
    pitch = float(orientation[2])
    yaw = float(orientation[3])
    rpy_vel_x = float(orientation[4])
    rpy_vel_y = float(orientation[5])
    rpy_vel_z = float(orientation[6])
    position_x = float(orientation[7]) # position of the SPINE
    position_y = float(orientation[8])
    position_z = float(orientation[9])
    vx = float(orientation[10])
    vy = float(orientation[11])
    vz = float(orientation[12])
    sim_time = float(orientation[13])
    hip_l_z = float(orientation[14])
    hip_l_x = float(orientation[15])
    hip_l_y = float(orientation[16])
    knee_l = float(orientation[17])
    ankle_l_y = float(orientation[18])
    ankle_l_x = float(orientation[19])
    hip_r_z = float(orientation[20])
    hip_r_x = float(orientation[21])
    hip_r_y = float(orientation[22])
    knee_r = float(orientation[23])
    ankle_r_y = float(orientation[24])
    ankle_r_x = float(orientation[25])
    adj_x = float(orientation[26]) # These are the positions that we're interested in (min/max or avg foot location)
    adj_y = float(orientation[27])
        
    print('fallen_status:{} roll:{} pitch:{} yaw:{} rpy_vel_x:{} rpy_vel_y:{} rpy_vel_z:{} position_x:{} position_y:{} position_z:{} vx:{} vy:{} vz:{} sim_time:{} adj_x:{}, adj_y:{}'.format(
            fallen_status, roll, pitch, yaw, rpy_vel_x, rpy_vel_y, rpy_vel_z, position_x, position_y, position_z, vx, vy, vz, sim_time, adj_x, adj_y))
    
    # Update joints to actual joint states (received from ros)
    # Negative numbers ARE possible, since we're dealing with actual joint positions, not theoretical ones.
    # TODO: consider using true_joint_states instead of subtracting min
    joint_states = [hip_l_x - hip_x_min_temp, hip_l_y - hip_y_min_temp, knee_l - knee_min_temp, ankle_l_y - ankle_y_min_temp, ankle_l_x - ankle_x_min_temp, 
                    hip_r_x - hip_x_min_temp, hip_r_y - hip_y_min_temp, knee_r - knee_min_temp, ankle_r_y - ankle_y_min_temp, ankle_r_x - ankle_x_min_temp,
                    hip_l_z - hip_z_min_temp, hip_r_z - hip_z_min_temp] # All positives for relu
    #joint_states means adjusted joint states, not ideal joint states! :)
    print('joint_states:{}'.format(joint_states))
    true_joint_states = [hip_l_x, hip_l_y, hip_l_z, knee_l, ankle_l_y, ankle_l_x,
                    hip_r_x, hip_r_y, hip_r_z, knee_r, ankle_r_y, ankle_r_x, 
                    ball_l, 
                    ball_r, 
                    spine_x, spine_y, spine_z]

    
    words = msg.split(' ')

    distance_new = float(words[1].split(':')[1])
    print('orig_x:{} orig_y:{}'.format(position_x, position_y)) # 
    position_x = adj_x
    position_y = adj_y
    print('adjusted_x:{} adjusted_y:{}'.format(position_x, position_y))
    # TODO: return actual x and y once trained well enough
    return(State(joint_states, [roll, pitch, yaw], [rpy_vel_x, rpy_vel_y, rpy_vel_z], [position_x, position_y, position_z], [vx, vy, vz], true_joint_states, distance_new), fallen_status, distance_new, roll, pitch, yaw, duration, sim_time)


def calculate_score(distance_old, distance_new):
    return(distance_new - distance_old)


    

