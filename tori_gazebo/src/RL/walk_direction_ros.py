import sys
from utils_walk_vanilla_ros import *
import classes
import numpy as np
import comm_test6
import random
import pickle
import time
import math
import TD3
import numpy as np
import matplotlib.pyplot as plt
from utils import plotLearning, ReplayBuffer
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from tori_msgs.msg import State, Replay, ToriJointAngles, Float # For some reason, ROS2 won't allow std_msgs. vision isn't working either, so this seems to be a path problem, especially after the message from using colcon build.

#joint_limits = [[[0, 2.36], [-1.57, 0.52], [-1.57, 1.57], [-2.618, 0.0], [-1.13, 0.44], [-1.57, 0.52]], [[0, 2.36], [-1.57, .52], [-1.57, 1.57], [-2.618, 0.0], [-1.13, 0.44], [-1.57, 0.52]], [[-0.52, 1.57]], [[-0.52, 1.57]], [[-0.52, 1.57], [-0.79, 0.79], [-1.57, 1.57]]]  # added soft limits in utils.py


class Walk(Node):
    def __init__(self):
        self.computer = 'kelsey'
        super().__init__('walk_node_{}'.format(self.computer)) #Remember to use ros_bridge
        self.pkl_folder = 'pkl_walk_direction'
        self.pause_on_nn = False
        self.score = 0
        self.reward = 0.0
        self.distance_new = 0.0
        self.distance_old = 0.0
        self.reward_total = 0.0
        self.joint_states_init = [0.0 - hip_x_min_temp, 0.0 - hip_y_min_temp, 0.0 - knee_min_temp, 0.0 - ankle_y_min_temp,  0.0 - ankle_x_min_temp, 0.0 - hip_x_min_temp, 0.0 - hip_y_min_temp, 0.0 - knee_min_temp, 0.0 - ankle_y_min_temp,  0.0 - ankle_x_min_temp, 0.0 - hip_z_min_temp, 0.0 - hip_z_min_temp]#, [0.0, 0.0, 0.0, 0.0]]#, [0.0], [0.0], [0.0, 0.0, 0.0]] 
        self.joint_states = self.joint_states_init
        self.true_joint_states_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.true_joint_states = self.true_joint_states_init
        self.score_hist = []
        self.distance_hist_long = []
        
        
        self.fallen_status = 0
        self.vel_init = [0, 0, 0]
        self.positions_init = [0, 0.80]  
        self.rpy_init = [0., 0., math.pi]
        self.rpy_vel_init = [0, 0, 0]
        self.y_pos_init = [0.0]
        self.s = self.s1 = classes.State(self.joint_states, self.rpy_init, self.rpy_vel_init, self.positions_init, self.vel_init, self.true_joint_states, self.y_pos_init)
        
        self.gamma = 0.99                # discount for future rewards
        self.batch_size = 128            # num of transitions sampled from replay buffer
        self.num_actions = 12
        self.action_init = [(hip_x_min_temp/(hip_x_min_temp-hip_x_max_temp)), (hip_y_min_temp/(hip_y_min_temp-hip_y_max_temp)), (knee_min_temp/(knee_min_temp-knee_max_temp)), (ankle_y_min_temp/(ankle_y_min_temp-ankle_y_max_temp)), (ankle_x_min_temp/(ankle_x_min_temp-ankle_x_max_temp)),
                           (hip_x_min_temp/(hip_x_min_temp-hip_x_max_temp)), (hip_y_min_temp/(hip_y_min_temp-hip_y_max_temp)), (knee_min_temp/(knee_min_temp-knee_max_temp)), (ankle_y_min_temp/(ankle_y_min_temp-ankle_y_max_temp)), (ankle_x_min_temp/(ankle_x_min_temp-ankle_x_max_temp)),
                           (hip_z_min_temp/(hip_z_min_temp-hip_z_max_temp)), (hip_z_min_temp/(hip_z_min_temp-hip_z_max_temp))]
        self.action_init = np.array([(x*2)-1 for x in self.action_init])
        
        self.exploration_noise_init = 0.08 #.10, 0.05
        self.exploration_noise = self.exploration_noise_init
        self.polyak = 0.995              # target policy update parameter (1-tau)
        self.policy_noise = 0.12 #.20, .10          # target policy smoothing noise
        self.noise_clip = 0.5
        self.policy_delay = 2            # delayed policy updates parameter
        
        self.testing = False
        if self.testing == True:
            self.policy_noise = 0.0
            self.exploration_noise_init = 0.0
            self.exploration_noise = 0.0
        self.last_saved_index = 150000 #minimax at 1165000; lowered effort to 3.92 and renamed i to 1000 at 1419000; instant at 8000
                                  # policy .05, exp .04 @ 137000; reverted noise to policy .12 and exp .08 @ 200000
        self.distance_hist = []
        
        if self.last_saved_index > 0:
            self.read_pkl = True
        else:
            self.read_pkl = False
        
        self.i = self.last_saved_index
        
        v = True
        self.j = 0
        lr = .0001
        self.num_states = 22
        remove_states = []
        load_weights = True
        read_replay_buffer = True
        add_num_states = 0
        add_actions = 0
        layer_height=250
        
        if self.read_pkl == True:
            
            #agent = NewAgent.load_model('./pkl/agent_{}.pkl'.format(i))
            
            if load_weights == True:
                print('reading weights...')
                self.agent = TD3.TD3(lr=lr, state_dim=self.num_states + add_num_states - len(remove_states), action_dim=self.num_actions + add_actions, max_action=1.0, layer_height=layer_height)
                
                self.agent.load('./{}'.format(self.pkl_folder), self.i, additional_dims=add_num_states, additional_actions=add_actions, remove_dimensions_=remove_states)
                self.num_actions = self.num_actions + add_actions
                #print('STATES:{}'.format(agent.))
                #agent.state_dim += add_num_states
                #if add_state > 0:
                    
            else:
                print('WARNING: LOADING FULL AGENT')
                self.agent = TD3.TD3.load_model('./{}/agent_{}.pkl'.format(self.pkl_folder, self.i))
                self.agent.use_scheduler = False
            if read_replay_buffer == True:
                print('reading replay buffer...')
                self.replay_buffer = pickle.load(open('./{}/replay_{}.pkl'.format(self.pkl_folder, self.i), 'rb'))
            else:
                self.replay_buffer = ReplayBuffer()
        else:
            print('creating agent')
            #agent = NewAgent(alpha=0.000005, beta=0.00001, input_dims=[3], gamma=1.01, layer1_size=30, layer2_size=30, n_outputs=1, n_actions=26) # 26=13*2
            #agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[19], tau=0.001, env='dummy', sigma=.5,
            #          batch_size=100,  layer1_size=200, layer2_size=250, n_actions=12, max_size=100000)
            self.agent = TD3.TD3(lr=lr, state_dim=self.num_states, action_dim=self.num_actions, max_action=1.0, layer_height=layer_height)
            self.replay_buffer = ReplayBuffer()
        self.state_sub = self.create_subscription(State, '/tori_state_{}'.format(self.computer), self.state_callback) # listens for state updates
        self.reward_sub = self.create_subscription(State, '/tori_state_{}'.format(self.computer), self.reward_callback) # 
        self.replay_sub = self.create_subscription(Replay, '/replay', self.replay_callback) # listens for replay messages
        self.replay_pub = self.create_publisher(Replay, '/replay', qos_profile=1) # publishes replay to this/other computers
        self.joint_angles_pub = self.create_publisher(ToriJointAngles, '/tori_joint_command_{}'.format(self.computer)) # tells control_motion the desired joint positions
        self.checkpoint_sub = self.create_subscription(Float64, '/checkpoint_{}'.format(self.computer), self.checkpoint_callback)
        self.checkpoit_pub = self.create_publisher(Float64, '/checkpoint_{}'.format(self.computer), qos_profile=0)
        
        # start training
        self.state_pub = self.create_publisher(State, '/tori_state_{}'.format(self.computer), qos_profile=1)
        state = State()
        state.fallen_status = float(self.fallen_status)
        state.orientation = self.rpy_init
        state.pos = [0., 0., 0.80] #TODO: get position_y_spine, not necessarily minimin
        state.distance_minimum = -.02 #TODO: check this number
        state.rpy_vel = [0., 0., 0.]
        state.vel = [0., 0., 0.]
        state.sim_time = 0.0
        self.state_pub.publish(state)
        print('published!')
        
    def state_callback(self, msg): # maybe somehow combine with reward callback? or make this a non-callback definition?
        pass    
        '''
        # when exactly would this be called?
        print('\nround {}'.format(i))
        print('j:{}'.format(j))
        # update current state
        #print('last action:{}'.format(action))
        s = classes.State(s1.joint_states, s1.rpy, s1.rpy_vel, s1.positions, s1.position_vel, s1.true_joint_states, s1.y_pos)
        
        print('s positions:{}'.format(s.positions))
        print('s velocities:{}'.format(s.position_vel))
        print('s rpy:{}'.format(s.rpy))
        print('s rpy_vel:{}'.format(s.rpy_vel))
    
        obs = np.array([s.positions[1]] + s.position_vel + s.rpy + s.rpy_vel + s.joint_states)
        #print('obs:{}'.format(obs))
        action = agent.select_action(obs)
        print('raw action:{}'.format(action))
        if testing == False:        
            exploration_noise = exploration_noise_init
            noise = np.random.normal(0, exploration_noise, size=num_actions)
            action = action + noise
            override = False
            if override == True:
                action[10] = max(min(action[10], .2), -.2) # "Clip" hip_z action between +-20%
                action[11] = max(min(action[11], .2), -.2)
            action = action.clip(-1.0, 1.0)
            print('clipped actions:{}'.format(action))
            print('exploration noise:{}'.format(exploration_noise))
        
    
        #print('ACTION:{}'.format(action))
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
        duration = .4
        print('duration:{}'.format(duration))
        tori_joint_angles = Tori_Joint_Angles()
        tori_joint_angles.angles = [hip_l_x, hip_l_y, hip_l_z, knee_l, ankle_l_y, ankle_l_x,
                            hip_r_x, hip_r_y, hip_r_z, knee_r, ankle_r_y, ankle_r_x, 
                            ball_l, 
                            ball_r, 
                            spine_x, spine_y, spine_z,
                            duration]
        
        #publish joint commands
        self.joint_angles_pub.publish(tori_joint_angles)
        '''

    def reward_callback(self, msg): #receive state from control_motion and calculate reward
        if self.i > self.last_saved_index:
            self.get_logger().info('I heard truncated:{}'.format(data.header))
            fallen_status = msg.fallen_status
            roll, pitch, yaw = msg.orientation
            rpy_vel_x, rpy_vel_y, rpy_vel_z = msg.rpy_vel
            pos_x, pos_y, pos_z = msg.pos #TODO: y_pos is different than distance!!! distance should be minimin
            vx, vy, vz = msg.vel
            sim_time = msg.sim_time
            
            hip_l_z, hip_l_x, hip_l_y, knee_l, ankle_l_y, ankle_l_x, hip_r_z, hip_r_x, hip_r_y, knee_r, ankle_r_y, ankle_r_x = msg.joint_positions
            
            joint_states = [hip_l_x - hip_x_min_temp, hip_l_y - hip_y_min_temp, knee_l - knee_min_temp, ankle_l_y - ankle_y_min_temp, ankle_l_x - ankle_x_min_temp, 
                            hip_r_x - hip_x_min_temp, hip_r_y - hip_y_min_temp, knee_r - knee_min_temp, ankle_r_y - ankle_y_min_temp, ankle_r_x - ankle_x_min_temp,
                            hip_l_z - hip_z_min_temp, hip_r_z - hip_z_min_temp]
            
            distance_old = distance_new
            distance_new = msg.distance_minimin
            s1 = State(joint_states, [roll, pitch, yaw], [rpy_vel_x, rpy_vel_y, rpy_vel_z], [position_x, position_z], [vx, vy, vz], true_joint_states, distance_new) 
            #true joint states is a misnomer. It means joint_position, where lower_limit < joint_position < upper_limit
    
            distance_hist.append(distance_new)
            reward = 0.0
        
            print('ROLL:{} PITCH:{} YAW:{}'.format(abs(roll), abs(pitch), abs(yaw)))
        
            flat_bonus = 0
            roll_bonus = 0 #3 * abs(np.tanh(.05/roll))
            try:
                pitch_bonus = .1 * abs(np.tanh(.04/pitch))
                if pitch_bonus > 1:
                    pitch_bonus = 1.
            except:
                pitch_bonus = 1.
            yaw_bonus = -.2 * abs(np.tanh(math.pi - abs(yaw))) #-.2 * abs(np.tanh(math.pi - abs(yaw)))
            
            time_bonus = .1 #.01
        
            distance_bonus = 0
        
            distance_bonus = 10.0 * (distance_new - distance_old)
        
            # All rewards shuold be continuous and differentiable
            # Rewards should not be cumulative (ex. reward=1 for lasting 1 second, reward=5 for lasting 5 seconds, etc) or this will throw off the reward estimator
            reward = distance_bonus + pitch_bonus + yaw_bonus + time_bonus
            reward_total += reward
            print(s1.y_pos)
            obs_ = np.array([s1.positions[1]] + s1.position_vel + s1.rpy + s1.rpy_vel + s1.joint_states)# + s1.y_pos)# + list(action))
            
            replay = Replay()
            replay.obs = obs
            replay.action = action
            replay.reward = reward
            replay.obs_ = obs_
            replay.fallen_status = float(fallen_status) #TODO: differentiate between fallen & terminal
            self.replay_pub.publish(replay)
            
            reward_total = 0.0
            
            score += reward  
        
            if v == True:
                print('distance_bonus:{}'.format(distance_bonus))
                print('pitch_bonus:{}'.format(pitch_bonus))
                print('yaw_bonus:{}'.format(yaw_bonus))
                print('reward:{}'.format(reward))
                print('reward_total:{}'.format(reward_total))
                print('score:{}'.format(score))
            if fallen_status == 1:
                # if the robot falls, s1 is automatically back to neutral position, so we'll update all states to also be neutral
                if testing == False:
                    if i > last_saved_index + 0:
                        pass
                        #this may be unnecessary, since we're updating in replay_callback each iteration
                        #agent.update(replay_buffer, j, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                    
                distance_hist_long.append(max(distance_hist))
                distance_new = distance_old = distance = -.02 #TODO: check this value
                
                #action = action_init
                distance_hist = []
                s.rpy = rpy_init
                s.rpy_vel = rpy_vel_init
                s.positions = positions_init
                s.position_vel = vel_init
                s.joint_states = joint_states_init#, [0.0, 0.0, 0.0, 0.0]]
                s.true_joint_states = true_joint_states_init
                s.y_pos = y_pos_init
                s1 = s
                j = 0
                reward_total = 0.0
                score_hist.append(score)
                score = 0
                    # when exactly would this be called?
        print('\nround {}'.format(self.i))
        print('j:{}'.format(self.j))
        # update current state
        #print('last action:{}'.format(action))
        self.s = classes.State(self.s1.joint_states, self.s1.rpy, self.s1.rpy_vel, self.s1.positions, self.s1.position_vel, self.s1.true_joint_states, self.s1.y_pos)
        
        print('s positions:{}'.format(self.s.positions))
        print('s velocities:{}'.format(self.s.position_vel))
        print('s rpy:{}'.format(self.s.rpy))
        print('s rpy_vel:{}'.format(self.s.rpy_vel))
    
        obs = np.array([self.s.positions[1]] + self.s.position_vel + self.s.rpy + self.s.rpy_vel + self.s.joint_states)
        #print('obs:{}'.format(obs))
        action = self.agent.select_action(obs)
        print('raw action:{}'.format(action))
        if self.testing == False:        
            exploration_noise = self.exploration_noise_init
            noise = np.random.normal(0, exploration_noise, size=self.num_actions)
            action = action + noise
            override = False
            if override == True:
                action[10] = max(min(action[10], .2), -.2) # "Clip" hip_z action between +-20%
                action[11] = max(min(action[11], .2), -.2)
            action = action.clip(-1.0, 1.0)
            print('clipped actions:{}'.format(action))
            print('exploration noise:{}'.format(exploration_noise))
        
    
        #print('ACTION:{}'.format(action))
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
        
        ball_l = 0. #(ball_l_max - ball_l_min) * adj_actions[10] + ball_l_min
        ball_r = 0. #(ball_r_max - ball_r_min) * adj_actions[11] + ball_r_min
        
        spine_x = 0.
        spine_y = 0.
        spine_z = 0.
        duration = .4
        print('duration:{}'.format(duration))
        tori_joint_angles = ToriJointAngles()
        tori_joint_angles.angles = [hip_l_x, hip_l_y, hip_l_z, knee_l, ankle_l_y, ankle_l_x,
                            hip_r_x, hip_r_y, hip_r_z, knee_r, ankle_r_y, ankle_r_x, 
                            ball_l, 
                            ball_r, 
                            spine_x, spine_y, spine_z]
        tori_joint_angles.duration = duration
        
        #publish joint commands
        self.joint_angles_pub.publish(tori_joint_angles)
        self.i += 1
        self.j += 1 # TODO: maybe move these higher?
            
    def replay_callback(self, msg):
        replay_buffer.add((msg.obs, msg.action, msg.reward, msg.obs_, msg.fallen_status)) #discount factor already takes future rewards into account!
        agent.update(replay_buffer, 1, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
    
    def checkpoint_callback(self, msg): # dummy msg
        if i % 1000 == 0 and i != last_saved_index:
            exploration_noise = exploration_noise_init
            print('sending pause')
            server.send_message('pause')
            #TODO: PAUSE!!!
            if i % 1000 == 0:
                agent.save_model('./{}/agent_{}.pkl'.format(pkl_folder, i))
                agent.save('./{}'.format(pkl_folder), i)
                pickle.dump(replay_buffer, open('./{}/replay_{}.pkl'.format(pkl_folder, i), 'wb'))
                
            filename = './{}/TD3_{}.png'.format(pkl_folder, i)
            plotLearning(score_hist, filename=filename, window=5, erase=True)
                        
            filename = './{}/distance_hist_{}.png'.format(pkl_folder, i)
            plotLearning(distance_hist_long, filename=filename, window=5, erase=True, ylabel_='Distance (m)')
            #TODO: UNPAUSE!!!
    
rclpy.init()
walk_node = Walk()
rclpy.spin(walk_node)
walk_node.destroy_node()
rclpy.shutdown()