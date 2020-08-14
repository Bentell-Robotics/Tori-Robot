import sys
from walk_direction_utils import *
import classes
import numpy as np
import comm_test6
import random
import pickle
import time
import math
from math import pi, cos, sin, sqrt
import TD3
import numpy as np
import matplotlib.pyplot as plt
from utils import plotLearning, ReplayBuffer
import sympy as sp

#joint_limits = [[[0, 2.36], [-1.57, 0.52], [-1.57, 1.57], [-2.618, 0.0], [-1.13, 0.44], [-1.57, 0.52]], [[0, 2.36], [-1.57, .52], [-1.57, 1.57], [-2.618, 0.0], [-1.13, 0.44], [-1.57, 0.52]], [[-0.52, 1.57]], [[-0.52, 1.57]], [[-0.52, 1.57], [-0.79, 0.79], [-1.57, 1.57]]]  # added soft limits in utils.py
pkl_folder = 'pkl_walk_direction'

score = 0
reward = 0.0
distance_new = 0.0
distance_old = 0.0
joint_states_init = [0.0 - hip_x_min_temp, 0.0 - hip_y_min_temp, 0.0 - knee_min_temp, 0.0 - ankle_y_min_temp,  0.0 - ankle_x_min_temp, 0.0 - hip_x_min_temp, 0.0 - hip_y_min_temp, 0.0 - knee_min_temp, 0.0 - ankle_y_min_temp,  0.0 - ankle_x_min_temp, 0.0 - hip_z_min_temp, 0.0 - hip_z_min_temp]#, [0.0, 0.0, 0.0, 0.0]]#, [0.0], [0.0], [0.0, 0.0, 0.0]] 
joint_states = joint_states_init
true_joint_states_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
true_joint_states = true_joint_states_init
score_hist = []
distance_hist_long = []
game_avg = []


fallen_status = 0
vel_init = [0, 0, 0]
positions_init = [0.0, 0.0, 0.80]  
rpy_init = [0, 0, -1.0, 0.0]
rpy_vel_init = [0, 0, 0]
y_pos_init = [0.0]

max_rad = 2*pi
min_rad = 0
#sign = 1 if random.random() < .5 else -1
target_yaw = random.random() * pi/2 + pi/4 # * sign
#while min_rad < target_yaw < max_rad:
#    target_yaw = random.random() * pi * sign

use_avg_foot_location = True
s = s1 = classes.State(joint_states, rpy_init, rpy_vel_init, positions_init, vel_init, true_joint_states, y_pos_init)
#if use_avg_foot_location == True:
    
if use_avg_foot_location == False:
    if pi/2 < target_yaw < 3*pi/2: # goal on left
        s.positions[0] = .1 # initial x_min
        s1.positions[0] = .1
    else:
        s.positions[0] = -.1
        s1.positions[0] = -.1
gamma = 0.99                # discount for future rewards
batch_size = 128            # num of transitions sampled from replay buffer
num_actions = 12
#action_init = [(hip_x_min_temp/(hip_x_min_temp-hip_x_max_temp)), (hip_y_min_temp/(hip_y_min_temp-hip_y_max_temp)), (knee_min_temp/(knee_min_temp-knee_max_temp)), (ankle_y_min_temp/(ankle_y_min_temp-ankle_y_max_temp)), (ankle_x_min_temp/(ankle_x_min_temp-ankle_x_max_temp)),
#                   (hip_x_min_temp/(hip_x_min_temp-hip_x_max_temp)), (hip_y_min_temp/(hip_y_min_temp-hip_y_max_temp)), (knee_min_temp/(knee_min_temp-knee_max_temp)), (ankle_y_min_temp/(ankle_y_min_temp-ankle_y_max_temp)), (ankle_x_min_temp/(ankle_x_min_temp-ankle_x_max_temp)),
#                   (hip_z_min_temp/(hip_z_min_temp-hip_z_max_temp)), (hip_z_min_temp/(hip_z_min_temp-hip_z_max_temp))]
#action_init = np.array([(x*2)-1 for x in action_init])

exploration_noise_init = 0.10 #.10, 0.05
exploration_noise = exploration_noise_init
polyak = 0.995              # target policy update parameter (1-tau)
policy_noise = 0.20 #.20, .10          # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2            # delayed policy updates parameter
her = False
adv = True # advanced version of her
adv_obs = []
adv_obs_ = []
adv_reward = []
adv_action = []
adv_pos = [[0.0, 0.0]]
imagine = False
reward_noise = 0.0 #.20

last_action = []
time.sleep(2)
testing = False
if testing == True:
    policy_noise = 0.0
    exploration_noise_init = 0.0
    exploration_noise = 0.0
last_saved_index = 534000 # raised exp policy to from .05 .05 to .10 .20 @ 10000
                          # corrected distance equation and increased time reward from .1 to 1 @ 108000
                          # implemented her @ 157000; 0 distance_bonus when fallen @ 160000
                          # implemented imagine @ 251000 (obsolete)
                          # corrected imagine @ 259000 (obsolete)
                          # set her and adv True @ 272000
                          # hip_y min from -.22 to -.52, ankle_y max from .22 to .52 @ 274000 (ckpt) (ckpt)
                          # increased yaw_penalty from .2 to 2 @ 351000 (accidental override @ 279000) (undid increase & abs() change)
                          # yaw bonus always 0 @ 351000 (ckpt)
                          # fixed distance measurement @ 357000
                          # corrected her_yaw @ 372000
                          # fixed distance bonus persistently 0 @ 357000
                          # corrected distance again @ 351000
                          # added her again @ 359000
                          # increased exp from .10 to .20 @ 362000
                          # started from pretrained model and reverted joints to -.22 & .22 hip_y min ankle_y max resp
                          # fixed repetitive sympy import location @ 101000
                          # disabled her @ 132000
                          # tested device=cpu @211000-213000
                          # trained extra 5000 @ 240000
                          # lowered exp, policy to .02 @ 242000
                          # raised exp to .20, inc range from pi/2+pi/4 to pi+0
                          # lowered exp to .10 @ 328000
                          # decomposed yaw @ 358000
                          # exp 10 policy 10 to exp 15 policy 15 @ 361000
                          # target_yaw pi/2 @ 487000 (ckpt)
'''
                          # exp .10 policy .10 to exp .20 policy .20 @ 520000
                          # exp .20 policy .20 to exp .10 policy .10 @ 540000
                          # lr .0001 to .001 @ 594000 (before this was changed, it kept tripping after 2 steps all night...)
                          # exp .10 to .20, read replay false @ 598000
                          # exp, policy to .25 @ 600000
'''
                          # replaced delta_pos with pos for distance, widened range from just pi/2, to 0 to pi @ 509000
                          # shortened range to ip/2 + pi/4, raised lr from .001 to .01 @ 514000 (learned to go backwards due to high reward for some reason... lowered lr to .005)
                          # time bonus 0 @ 615000
                          # lowered learning rate from .005 to .001 @ 637000
                          # adv True, lr .0005 @ 650000
                          # adv False, min distance feet & torso_lower @ 664000
                          # lowered policy noise .15 to .20, lr .001 @ 679000
                          # exp .05 policy .10 @ 688000
                          # exp .20 @ 764000
                          # hip_z +-.78 to +-.4, exp .15 @ 766000
                          # q = self.l5(q)!!! previously was q = self.l3(q), removed one hidden layer, started agent fresh @ 0
                          # lr .0002 @ 3000
                          # exp .20 policy .20 @ 4000
                          # exp .40, read False @ 8000
                          # took replay from 766000, exp .20 @ 15000
                          # lr .001 @ 16000
                          # increasing exp (.10 -> .20) @ 33000
                          # pi/2 * random() + pi/4 @ 144000
                          # adv True @ 152000
                          # average foot location @ 171000
                          # exp .10, fixed adv distance not updating (never cleared adv_pos, so it kept grabbing positions from the start of training) @ 201000
                          # hip_z min/max from +-.4 to +-.6, lr .005 @ 292000 (before it kept sidestepping rather than turning its body) (ckpt)
'''
                          # lr .001 @ 394000 (kept taking very large steps and distance never increased, but maybe that's a result of avg foot distance)
'''
'''
                          # use_avg False @ 292000 
                          # adv False @ 293000
                          # lr .001 @ 321000 
'''
                          # iterations has a large effect on distance. set to 450 @ 292000 (read replay False)
                          # lr .002, exp .05 (<5) .10 (else) @ 330000
                          # lr .005, exp .10, hip_x/knee/ankle_x max/min/max +-.8 to +- .4, duration .3 to .2 @ 332000
                          # lr .002, exp .05 @ 375000
                          # Lowering duration rate requires fresh replay buffer
                          # duration rate .2 to .1, iter 500 @ 392000
                          # lr .0005, exp .02 @ 498000


distance_hist = []
game_hist = []
transfer_learning = False
if last_saved_index > 0 or transfer_learning == True:
    read_pkl = True
else:
    read_pkl = False

i = last_saved_index

v = True
j = 0
lr = .0005
num_states = 25
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
    #TODO: max size may be too low for transfer learning?
    '''
    new_rb = ReplayBuffer()
    for num, item in enumerate(replay_buffer.buffer):
        if num % 10000 == 0:
            print('original:{}'.format(item))
            line0 = np.append(replay_buffer.buffer[num][0], [pi])
            line12 = replay_buffer.buffer[num][1:3]
            line3 = np.append(replay_buffer.buffer[num][3], [pi])
            line4 = replay_buffer.buffer[num][4]
            line = np.empty(0)
            line = np.array(np.append(line, line0))
            line = np.array(np.append(line, line12))
            line = np.array(np.append(line, line3))
            line = np.array(np.append(line, line4))
            #new_rb.add(line)
            print('line:{}'.format(np.array(line)))
    '''
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
    extra_update = 0
extra_update = 0
additional_her_update = 0
print('connecting...')
server = comm_test6.Server(55430)

def solve_equation(xr, yr, mr, mt):
    #print('xr:{} yr:{} mr:{} mt:{}'.format(xr, yr, mr, mt))
    #sp.init_printing()
    x,y = sp.symbols('x,y')
    b = sp.symbols('b')
    h = sp.Eq(yr, mr * xr + b)
    br = sp.solve([h], (b))[b]
    
    f = sp.Eq(y, mt*x + 0)
    g = sp.Eq(y, mr*x + br)
    
    solution = sp.solve([f,g],(x,y))
    xd = solution[x]
    yd = solution[y]
    
    d = sqrt(xd**2 + yd**2) # distance along line from start when bt = 0
    print('br:{} xd:{} yd:{}'.format(br, xd, yd)) # doesn't care about positive or negitive!
    return(d)

#sym.solve([f,g,h],(x,y,z))
adjustment_points = [0]
exp_points = [.10]
policy_points = [.10]
adj_counter = 0
while True:
    if i in adjustment_points:
        exploration_noise_init = exp_points[adj_counter]
        exploration_noise = exp_points[adj_counter]
        adj_counter += 1
    
    print('\nround {}'.format(i))
    print('j:{}'.format(j))
    print('target_yaw:{}'.format(target_yaw))
    if pi/2 < target_yaw < 3*pi/2:
        print('LEFT')
    else:
        print('RIGHT')
    target_yaw_decomposed = [float(cos(target_yaw)), float(sin(target_yaw))]
    # update current state
    #print('last action:{}'.format(action))
    s = classes.State(s1.joint_states, s1.rpy, s1.rpy_vel, s1.positions, s1.position_vel, s1.true_joint_states, s1.y_pos)
    #print('s rpy original:{}'.format(s.rpy))
    #s.rpy.append(float(sin(s.rpy[2]))) # decompose yaw
    #s.rpy[2] = float(cos(s.rpy[2]))
    print('s positions:{}'.format(s.positions))
    #print('s velocities:{}'.format(s.position_vel))
    #print('s rpy:{}'.format(s.rpy))
    #print('s rpy_vel:{}'.format(s.rpy_vel))

    obs = np.array([s.positions[2]] + s.position_vel + s.rpy + s.rpy_vel + s.joint_states + target_yaw_decomposed)# + s.y_pos)#
    #print('obs:{}'.format(obs))
    action = agent.select_action(obs)
    #print('raw action:{}'.format(action))
    if testing == False:      
        if j < 5:
            exploration_noise = .2
        #elif j < 7:
        #    exploration_noise = .10
        else:
            exploration_noise = .2


        noise = np.random.normal(0, exploration_noise, size=num_actions)

        action = action + noise
        override = False
        if override == True:
            action[10] = max(min(action[10], .2), -.2) # "Clip" hip_z action between +-20%
            action[11] = max(min(action[11], .2), -.2)
        action = action.clip(-1.0, 1.0)
        #print('clipped actions:{}'.format(action))
        print('exploration noise:{}'.format(exploration_noise))
    

    #print('ACTION:{}'.format(action))
    s1, fallen_status, _, roll, pitch, yaw, duration, sim_time = new_state_after_action(s, action, target_yaw_decomposed, server, i, last_saved_index)  # new state after taking the best action
    print('s1 positions:{}'.format(s1.positions))
    s1.rpy.append(sin(s1.rpy[2])) # decompose yaw
    s1.rpy[2] = cos(s1.rpy[2])
    x_pos, y_pos, z_pos = s1.positions
    x_pos_prev, y_pos_prev, z_pos_prev = s.positions
    print('x_pos:{} y_pos:{}'.format(x_pos, y_pos))
    print('x_pos_prev:{} y_pos_prev:{}'.format(x_pos_prev, y_pos_prev))
    #x_pos, y_pos = adj_x, adj_y # use adjusted min/max coordinates
    adv_pos.append([x_pos, y_pos])
    delta_x = x_pos - x_pos_prev
    delta_y = y_pos - y_pos_prev
    if her == True:

        
        her_yaw = math.atan(delta_y / delta_x) #formerly just used actual yaw
        if delta_x < 0:
            her_yaw += pi
        obs_her = np.array([s.positions[2]] + s.position_vel + s.rpy + s.rpy_vel + s.joint_states + [her_yaw])

    #distance = sqrt((x_pos * cos(target_yaw - pi/2)) ** 2 + (y_pos * sin(target_yaw - pi/2)) ** 2)
    #distance_bonus = 10.0 * sqrt(((x_pos - x_pos_prev) * cos(target_yaw)) ** 2 + ((y_pos - y_pos_prev) * sin(target_yaw)) ** 2)
    slope_target = sin(target_yaw)/cos(target_yaw)
    slope_robot = -1 / slope_target
    #yd = solpe_target * xd + 0
    distance = solve_equation(x_pos, y_pos, slope_robot, slope_target) # get SCALAR distance from (0, 0)
    #distance = distance if delta_x/target_yaw_decomposed[0] >= 0 and delta_y/target_yaw_decomposed[1] >= 0 else -1 * distance
    if abs(target_yaw_decomposed[0]) < abs(target_yaw_decomposed[1]):
        distance = distance if y_pos/target_yaw_decomposed[1] >= 0 else -1 * distance #only applied to change in distance!
    elif abs(target_yaw_decomposed[0]) >= abs(target_yaw_decomposed[1]):
        distance = distance if x_pos/target_yaw_decomposed[0] >= 0 else -1 * distance


    # theta=pi/2 + n*pi ==> delta_x <= 0 (This is not currently the case, but it should be! Direction should be considered in function BEFORE distance is calculated!)
    print('delta_x:{}'.format(delta_x))
    print('delta_y:{}'.format(delta_y))
    distance_old = distance_new
    distance_new = distance
    distance_bonus = 10 * (distance_new - distance_old)
    if her == True:
        #distance_bonus_her = 10.0 * sqrt(((x_pos - x_pos_prev) * cos(her_yaw - pi/2)) ** 2 + ((y_pos - y_pos_prev) * sin(her_yaw - pi/2)) ** 2)
        distance_bonus_her = 10 * sqrt(delta_x**2 + delta_y**2)
    
    if imagine == True:
        obs_imagine = []
        target_yaw_imagine = []
        distance_bonus_imagine = []
        reward_imagine = []
        for _ in range(10):
            sign = 1 if random.random() < .5 else -1
            target_yaw_imagine_ = random.random() * pi * sign
            while min_rad < target_yaw_imagine_ < max_rad:
                target_yaw_imagine_ = random.random() * pi * sign
            target_yaw_imagine.append(target_yaw_imagine_)
            obs_imagine.append(np.array([s.positions[2]] + s.position_vel + s.rpy + s.rpy_vel + s.joint_states + [target_yaw_imagine_]))
            distance_old_imagine = sqrt((x_pos_prev * cos(target_yaw_imagine_ - pi/2)) ** 2 + (y_pos_prev * sin(target_yaw_imagine_ - pi/2)) ** 2)
            distance_new_imagine = sqrt((x_pos * cos(target_yaw_imagine_ - pi/2)) ** 2 + (y_pos * sin(target_yaw_imagine_ - pi/2)) ** 2)
            distance_bonus_imagine = 10.0 * (distance_new_imagine - distance_old_imagine)
            if fallen_status == 1 and distance_bonus_imagine > 0:
                distance_bonus_imagine = 0
            yaw_bonus_imagine = -.2 * abs(np.tanh(abs(target_yaw_imagine_) - abs(yaw)))
            reward_imagine.append(distance_bonus_imagine + yaw_bonus_imagine)
        
    if adv == True:
        adv_obs.append(np.array([s.positions[2]] + s.position_vel + s.rpy + s.rpy_vel + s.joint_states + [0, 0])) # target_yaw will be determined once fallen == 1
        adv_action.append(action)
    #distance_old = distance_new
    #distance_new = distance
    distance_hist.append(distance)
    reward = 0.0

    #print('ROLL:{} PITCH:{} YAW:{}'.format(abs(roll), abs(pitch), abs(yaw)))

    flat_bonus = 0
    roll_bonus = 0 #3 * abs(np.tanh(.05/roll))
    try:
        pitch_bonus = .1 * abs(np.tanh(.04/pitch))
        if pitch_bonus > 1:
            pitch_bonus = 1.
    except:
        pitch_bonus = 1.
    '''
    if yaw/target_yaw >= 0:       
        yaw_bonus = -.2 * abs(np.tanh(abs(target_yaw - yaw))) # if same L/R direction as target, find difference
    else:
        yaw_bonus = -.2 * abs(np.tanh(abs(target_yaw - yaw))) # iff opposite L/R direction, 
    '''
    yaw_bonus = 0
    yaw_bonus_her = 0
    
    
    time_bonus = 0 #.01

    #distance_bonus = 0

    #distance_bonus = 10.0 * (distance_new - distance_old)
    #distance_bonus_her = 10.0 * (distance_new_her - distance_old_her)
    
    '''
    if distance_bonus > 0 and fallen_status == 1:
        distance_bonus = 0
    if her == True and distance_bonus_her > 0 and fallen_status == 1:
        distance_bonus_her = 0
    '''
    
    if adv == True:
        adv_reward.append(pitch_bonus + time_bonus) # we'll add the yaw and distance reward once we know the target angle
    # All rewards shuold be differentiable
    # Rewards should not be cumulative (ex. reward=1 for lasting 1 second, reward=5 for lasting 5 seconds, etc) or this will throw off the reward estimator
    reward = distance_bonus + pitch_bonus + yaw_bonus + time_bonus
    if her == True:
        reward_her = distance_bonus_her + pitch_bonus + yaw_bonus_her + time_bonus
    
    if imagine == True:
        obs_imagine_ = []
        for k in range(10):
            #print('reward_imagine_old:{}'.format(reward_imagine[k]))
            reward_imagine[k] += time_bonus + pitch_bonus
            #print('reward_imagine_new:{}'.format(reward_imagine[k]))
            obs_imagine_.append(np.array([s1.positions[2]] + s1.position_vel + s1.rpy + s1.rpy_vel + s1.joint_states + [target_yaw_imagine[k]]))
            replay_buffer.add((obs_imagine[k], action, reward_imagine[k], obs_imagine_[k], float(fallen_status)))
    
    
    #print(s1.y_pos)
    reward += random.random() * reward_noise - reward_noise/2 # add noise to reward
    obs_ = np.array([s1.positions[2]] + s1.position_vel + s1.rpy + s1.rpy_vel + s1.joint_states + target_yaw_decomposed)# + s1.y_pos)# + list(action))
    
    if adv == True:
        adv_obs_.append(np.array([s1.positions[2]] + s1.position_vel + s1.rpy + s1.rpy_vel + s1.joint_states + [0, 0]))
    
    replay_buffer.add((obs, action, reward, obs_, float(fallen_status))) #discount factor already takes future rewards into account!
    if her == True:
        obs_her_ = np.array([s1.positions[2]] + s1.position_vel + s1.rpy + s1.rpy_vel + s1.joint_states + [her_yaw])
        print('her dist_bonus:{}'.format(distance_bonus_her))
        replay_buffer.add((obs_her, action, reward_her, obs_her_, float(fallen_status)))
        #agent.update(replay_buffer, 1, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay) # update doubled when her active
    
    

    j += 1
    
    score += reward  

    if v == True:
        print('distance_bonus:{}'.format(distance_bonus))
        print('distance_total:{}'.format(distance))
        print('pitch_bonus:{}'.format(pitch_bonus))
        #print('yaw_bonus:{}'.format(yaw_bonus))
        print('reward:{}'.format(reward))
        if her == True:
            print('reward_her:{}'.format(reward_her))
        print('score:{}'.format(score))
    if fallen_status == 1:
        min_iters = 2
        if adv == True and j >= min_iters:
            # calculate adv_target_yaw by taking the angle of the position right before when it landed
            adv_target_x = adv_pos[-min_iters][0]
            adv_target_y = adv_pos[-min_iters][1] # score may seem way off when j ends at a low number, since its first step usually puts the robot far to the left
            adv_target_yaw = math.atan(adv_target_y / adv_target_x)#math.atan(s.positions[1] / s.positions[0])
            #math.atan(delta_y / delta_x)
            adv_score = 0
            distance_old_adv = distance_new_adv = distance_adv = 0.0
            if adv_target_x < 0:
                adv_target_yaw += pi
            adv_target_yaw_decomposed = [cos(adv_target_yaw), sin(adv_target_yaw)]
            print('adv_target:{}'.format(adv_target_yaw))
            print('adv_target_decomposed:{}'.format(adv_target_yaw_decomposed))
            print('adv_pos:{}'.format(adv_pos))
            #slope_target = sin(adv_target_yaw)/cos(adv_target_yaw)
            slope_target = adv_target_yaw_decomposed[1]/adv_target_yaw_decomposed[0]
            slope_robot = -1 / slope_target  
            if use_avg_foot_location == False:
                # cannot use adv with use_avg False because it doesn't previously know whether to take the max or min foot positions :(
                if pi/2 < adv_target_yaw < 3*pi/2: # goal on left
                    adv_pos[0] = [.1, 0.0]
                else:
                    adv_pos[0] = [-.1, 0.0]
            for k in range(j):
                adv_fallen = 0
                adv_obs[k][-2] = adv_target_yaw_decomposed[0]
                adv_obs[k][-1] = adv_target_yaw_decomposed[1]
                adv_obs_[k][-2] = adv_target_yaw_decomposed[0]
                adv_obs_[k][-1] = adv_target_yaw_decomposed[1]
                #print('obs[k][-2]:{} obs[k][-1]:{} obs_[k][-2]:{} obs[k][-1]:{}'.format(adv_obs[k][-2], adv_obs[k][-1], adv_obs_[k][-2], adv_obs_[k][-1]))
                adv_x_prev = adv_pos[k][0]
                adv_y_prev = adv_pos[k][1]
                adv_x = adv_pos[k+1][0]
                adv_y = adv_pos[k+1][1]
                print('adv_x_prev:{} adv_y_prev:{} adv_x:{} adv_y:{}'.format(adv_x_prev, adv_y_prev, adv_x, adv_y))
                adv_reward[k] += 0#-.2 * abs(np.tanh(abs(abs(adv_target_yaw) - abs(adv_obs_[k][6]))))

                time_start = time.time()
                distance_adv = solve_equation(adv_x, adv_y, slope_robot, slope_target)
                #print('distance_adv_orig:{}'.format(distance_adv))
                
                #print('time:{}'.format(time.time() - time_start))

                delta_x_adv = adv_x - adv_x_prev
                delta_y_adv = adv_y - adv_y_prev
                #distance_adv = distance_adv if delta_x_adv/adv_target_yaw_decomposed[0] >= 0 and delta_y_adv/adv_target_yaw_decomposed[1] >= 0 else -1 * distance_adv
                if abs(adv_target_yaw_decomposed[0]) < abs(adv_target_yaw_decomposed[1]):
                    distance_adv = distance_adv if adv_y/adv_target_yaw_decomposed[1] >= 0 else -1 * distance_adv
                elif abs(adv_target_yaw_decomposed[0]) >= abs(adv_target_yaw_decomposed[1]):
                    distance_adv = distance_adv if adv_x/adv_target_yaw_decomposed[0] >= 0 else -1 * distance_adv
                
                distance_old_adv = distance_new_adv
                distance_new_adv = distance_adv
                distance_adv_bonus = 10 * (distance_new_adv - distance_old_adv)
                print('dist_adv_bonus:{}'.format(distance_adv_bonus))
                if k == j-1:
                    adv_fallen = 1
                    #distance_adv_bonus = 0
                adv_reward[k] += distance_adv_bonus
                #print('adv_reward:{}'.format(adv_reward[k]))
                adv_score += adv_reward[k]
                print('{}: adv_obs:{} adv_action:{} adv_reward:{} adv_obs_:{} adv_fallen:{}'.format(k, adv_obs[k], adv_action[k], adv_reward[k], adv_obs_[k], float(adv_fallen)))
                replay_buffer.add((adv_obs[k], adv_action[k], adv_reward[k], adv_obs_[k], float(adv_fallen))) # add to rb iff not fallen
            print(adv_score)
            adv_obs = []
            adv_obs_ = []
            adv_reward = []
            adv_action = []
            
            
        if testing == False:

            '''
            if imagine == True and her == True:
                j = j * 11
            '''
            if her == True and adv == True:
                j = j * 3
            elif her == True:
                j = j * 2
            elif adv == True:
                j = j * 2

            if i > last_saved_index + extra_update:
                agent.update(replay_buffer, j + extra_update + additional_her_update, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                extra_update = 0
                additional_her_update = 0
        
        distance_hist_long.append(max(distance_hist))
        game_hist.append(max(distance_hist))
        distance_new = distance_old = distance = 0
        #sign = 1 if random.random() < .5 else -1
        #target_yaw = random.random() * pi * sign
        target_yaw = random.random() * pi/2 + pi/4# + pi/2 + 0

        #action = action_init
        # if the robot falls, s1 is automatically back to neutral position, so we'll update all states to also be neutral

        distance_hist = []
        s.rpy = rpy_init
        s.rpy_vel = rpy_vel_init
        s.positions = positions_init
        if use_avg_foot_location == False:
    
            if pi/2 < target_yaw < 3*pi/2: # goal on left
                s.positions[0] = .1 # initial x_min
                adv_pos = [[50000, 50000]] # dummy, since we don't know which direction it lands yet
            else:
                s.positions[0] = -.1
                adv_pos = [[-50000, -50000]] 
        else:
            adv_pos = [[0.0, 0.0]] 
        s.position_vel = vel_init
        s.joint_states = joint_states_init#, [0.0, 0.0, 0.0, 0.0]]
        s.true_joint_states = true_joint_states_init
        s.y_pos = y_pos_init
        s1 = s
        j = 0
        score_hist.append(score)
        score = 0




    if i % 1000 == 0 and i != last_saved_index:
        game_avg.append(sum(game_hist)/len(game_hist))
        game_hist = []
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
        filename = './{}/game_avg_{}.png'.format(pkl_folder, i)
        plotLearning(game_avg, filename=filename, window=1, erase=True, xlabel_='1000 iters', ylabel_='Avg max distance (m)')


        #TODO: Record screen with multithreading
        server.send_message('unpause')
        print('sending unpause')
        time.sleep(0.002)

    i += 1

