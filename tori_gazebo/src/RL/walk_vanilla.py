import sys
from utils_walk_vanilla import *
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

#joint_limits = [[[0, 2.36], [-1.57, 0.52], [-1.57, 1.57], [-2.618, 0.0], [-1.13, 0.44], [-1.57, 0.52]], [[0, 2.36], [-1.57, .52], [-1.57, 1.57], [-2.618, 0.0], [-1.13, 0.44], [-1.57, 0.52]], [[-0.52, 1.57]], [[-0.52, 1.57]], [[-0.52, 1.57], [-0.79, 0.79], [-1.57, 1.57]]]  # added soft limits in utils.py
pkl_folder = 'pkl_walk_vanilla'

score = 0
reward = 0.0
distance_new = 0.0
distance_old = 0.0
reward_total = 0.0
joint_states_init = [0.0 - hip_x_min_temp, 0.0 - hip_y_min_temp, 0.0 - knee_min_temp, 0.0 - ankle_y_min_temp,  0.0 - ankle_x_min_temp, 0.0 - hip_x_min_temp, 0.0 - hip_y_min_temp, 0.0 - knee_min_temp, 0.0 - ankle_y_min_temp,  0.0 - ankle_x_min_temp, 0.0 - hip_z_min_temp, 0.0 - hip_z_min_temp]#, [0.0, 0.0, 0.0, 0.0]]#, [0.0], [0.0], [0.0, 0.0, 0.0]] 
joint_states = joint_states_init
true_joint_states_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
true_joint_states = true_joint_states_init
score_hist = []
distance_hist_long = []


fallen_status = 0
vel_init = [0, 0, 0]
positions_init = [0, 0.80]  
rpy_init = [0, 0, math.pi]
rpy_vel_init = [0, 0, 0]
y_pos_init = [0.0]
s = s1 = classes.State(joint_states, rpy_init, rpy_vel_init, positions_init, vel_init, true_joint_states, y_pos_init)

gamma = 0.99                # discount for future rewards
batch_size = 128            # num of transitions sampled from replay buffer
num_actions = 12
action_init = [(hip_x_min_temp/(hip_x_min_temp-hip_x_max_temp)), (hip_y_min_temp/(hip_y_min_temp-hip_y_max_temp)), (knee_min_temp/(knee_min_temp-knee_max_temp)), (ankle_y_min_temp/(ankle_y_min_temp-ankle_y_max_temp)), (ankle_x_min_temp/(ankle_x_min_temp-ankle_x_max_temp)),
                   (hip_x_min_temp/(hip_x_min_temp-hip_x_max_temp)), (hip_y_min_temp/(hip_y_min_temp-hip_y_max_temp)), (knee_min_temp/(knee_min_temp-knee_max_temp)), (ankle_y_min_temp/(ankle_y_min_temp-ankle_y_max_temp)), (ankle_x_min_temp/(ankle_x_min_temp-ankle_x_max_temp)),
                   (hip_z_min_temp/(hip_z_min_temp-hip_z_max_temp)), (hip_z_min_temp/(hip_z_min_temp-hip_z_max_temp))]
action_init = np.array([(x*2)-1 for x in action_init])

exploration_noise_init = 0.05 #.10, 0.05
exploration_noise = exploration_noise_init
polyak = 0.995              # target policy update parameter (1-tau)
policy_noise = 0.05 #.20, .10          # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2            # delayed policy updates parameter

last_action = []
time_bonus = 1
time.sleep(2)
testing = False
if testing == True:
    policy_noise = 0.0
    exploration_noise_init = 0.0
    exploration_noise = 0.0
last_saved_index = 433000 #minimax at 1165000; lowered effort to 3.92 and renamed i to 1000 at 1419000; instant at 8000
                          # policy .05, exp .04 @ 137000; reverted noise to policy .12 and exp .08 @ 200000
                          # raised exp .10 policy .18 @ 232000
                          # increased layer height to 250 @ 0; 50000 extra_update @ 8000; 175000 extra update @ 11000
                          # changed duration from .4 to .3 @219000 (progress was very slow)
                          # exp .05 policy .10 @ 409000
                          # lowered replay buffer size from 5e5 to 5e4, lowered policy to .05 @ 433000
distance_hist = []

if last_saved_index > 0:
    read_pkl = True
else:
    read_pkl = False

i = last_saved_index

v = True
j = 0
lr = .0001
num_states = 22
remove_states = []
load_weights = True
read_replay_buffer = True
add_num_states = 0
add_actions = 0
layer_height=250

if read_replay_buffer == True:
    print('reading replay buffer...')
    replay_buffer = pickle.load(open('./{}/replay_{}.pkl'.format(pkl_folder, i), 'rb'))
    replay_buffer.max_size = 5e4
else:
    replay_buffer = ReplayBuffer()
if read_pkl == True:
    
    #agent = NewAgent.load_model('./pkl/agent_{}.pkl'.format(i))
    
    if load_weights == True:
        print('reading weights...')
        agent = TD3.TD3(lr=lr, state_dim=num_states + add_num_states - len(remove_states), action_dim=num_actions + add_actions, max_action=1.0, layer_height=layer_height)
        
        agent.load('./{}'.format(pkl_folder), i, additional_dims=add_num_states, additional_actions=add_actions, remove_dimensions_=remove_states)
        num_actions = num_actions + add_actions
        #print('STATES:{}'.format(agent.))
        #agent.state_dim += add_num_states
        #if add_state > 0:
            
    else:
        print('WARNING: LOADING FULL AGENT')
        agent = TD3.TD3.load_model('./{}/agent_{}.pkl'.format(pkl_folder, i))
        agent.use_scheduler = False
else:
    print('creating agent')
    #agent = NewAgent(alpha=0.000005, beta=0.00001, input_dims=[3], gamma=1.01, layer1_size=30, layer2_size=30, n_outputs=1, n_actions=26) # 26=13*2
    #agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[19], tau=0.001, env='dummy', sigma=.5,
    #          batch_size=100,  layer1_size=200, layer2_size=250, n_actions=12, max_size=100000)
    agent = TD3.TD3(lr=lr, state_dim=num_states, action_dim=num_actions, max_action=1.0, layer_height=layer_height)
    #replay_buffer = ReplayBuffer()
    # may need to increase layer2_size
extra_update = 0
if extra_update > 0:
    print('updating...')
    agent.update(replay_buffer, extra_update, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
print('connecting...')
server = comm_test6.Server(55430)
while True:
    print('\nround {}'.format(i))
    print('j:{}'.format(j))
    # update current state
    #print('last action:{}'.format(action))
    s = classes.State(s1.joint_states, s1.rpy, s1.rpy_vel, s1.positions, s1.position_vel, s1.true_joint_states, s1.y_pos)
    
    print('s positions:{}'.format(s.positions))
    print('s velocities:{}'.format(s.position_vel))
    print('s rpy:{}'.format(s.rpy))
    print('s rpy_vel:{}'.format(s.rpy_vel))

    obs = np.array([s.positions[1]] + s.position_vel + s.rpy + s.rpy_vel + s.joint_states)# + s.y_pos)#
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
    s1, fallen_status, distance, roll, pitch, yaw, duration, sim_time = new_state_after_action(s, action, server, i, last_saved_index)  # new state after taking the best action

    distance_old = distance_new
    distance_new = distance
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
    
    replay_buffer.add((obs, action, reward, obs_, float(fallen_status))) #discount factor already takes future rewards into account!
    reward_total = 0.0
    
    

    j += 1

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
                agent.update(replay_buffer, j, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
            
        distance_hist_long.append(max(distance_hist))
        distance_new = distance_old = distance = -.02
        
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




    if i % 1000 == 0 and i != last_saved_index:
        exploration_noise = exploration_noise_init
        print('sending pause')
        server.send_message('pause')
        if i % 1000 == 0:
            agent.save_model('./{}/agent_{}.pkl'.format(pkl_folder, i))
            agent.save('./{}'.format(pkl_folder), i)
            pickle.dump(replay_buffer, open('./{}/replay_{}.pkl'.format(pkl_folder, i), 'wb'))
        filename = './{}/TD3_{}.png'.format(pkl_folder, i)
        plotLearning(score_hist, filename=filename, window=5, erase=True)
        
        #filename = './pkl/reward_recent_{}.png'.format(i)
        #reward_hist_recent = reward_hist[-1000:]
        #plotLearning(reward_hist, filename=filename, window=5, erase=True)
        #reward_hist = []
        
        filename = './{}/distance_hist_{}.png'.format(pkl_folder, i)
        #reward_hist_recent = reward_hist[-1000:]
        plotLearning(distance_hist_long, filename=filename, window=5, erase=True, ylabel_='Distance (m)')
        #distance_hist = []
        '''
        try:
            for a in range(len(action_hist)):
                for k in range(len(action_hist[a][0])):
                    filename = './pkl/action_{}_{}.png'.format(a, i)
                    if k+1 == len(action_hist[a][0]):
                        erase = True
                    else:
                        erase = False
                    plotLearning([action_hist[a][x][k] for x in range(len(action_hist[a]))], filename=filename, window=1, erase=erase)
        except:
            with open('./problem.txt', 'w') as outfile:
                outfile.write(str(action_hist) + '\na:{}\nk:{}\n'.format(a, k))
        '''
        #TODO: Record screen with multithreading
        server.send_message('unpause')
        print('sending unpause')
        time.sleep(0.002)

    i += 1

