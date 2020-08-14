#!/usr/bin/env python
import os
import rospy
import comm_test5
import roslib; roslib.load_manifest('tori_gazebo')
import tf
from gazebo_msgs.srv import *
import time
from std_msgs.msg import Float64
import control_msgs.msg # for action interface (not used yet)
import trajectory_msgs.msg # for topic interface for joint_trajectory_controller
#std_msgs.msg.
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty, EmptyRequest
#import interface
import std_srvs
import geometry_msgs.msg
import time
from tf.transformations import euler_from_quaternion
#geometry_msgs.msg.Twist.linear.

# First set up the move group interface
#mgpi = interface.MoveGroupPythonInteface()
#groups = [mgpi.leg_l_group, mgpi.leg_r_group, mgpi.toes_l_group, mgpi.toes_r_group, mgpi.spine_group]


#client = client_py3.Client()

class JTPoint(object):
    def __init__(self, point_number, num_points, now, duration, starts, joint_intervals, instant=False): # formerly nsecs_interval instead of duration
        if instant == False:
            duration_float = duration[0] + duration[1]/1000000000.
        
            interval = duration_float / (num_points-1)
        elif instant == True:
            #print('Divide by zero; Setting interval to 0')
            interval = 0.016 # use a small number to prevent ROS from dropping point
        self.point = trajectory_msgs.msg.JointTrajectoryPoint()
        self.point.positions = []
        for i in range(len(starts)):
            self.point.positions.append(starts[i] + joint_intervals[i]*point_number)
        current_point_time = now.secs + now.nsecs/1000000000. + interval*point_number
        secs, nsecs = int(current_point_time), (float(current_point_time) - int(current_point_time)) * 1000000000
        self.point.time_from_start.set(int(secs), int(nsecs))
        #print('point_num:{} secs:{} nsecs:{}'.format(point_number, secs, nsecs))

def reset_simulation():
    rospy.init_node('reset_world')
    rospy.wait_for_service('/gazebo/reset_world')
    reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    reset_world()
    
def pause_simulation():
    pause_physics_client=rospy.ServiceProxy('/gazebo/pause_physics',Empty)
    pause_physics_client.call()
    
def unpause_simulation():
    unpause_physics_client=rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
    unpause_physics_client.call()

def gms_client(model_name,relative_entity_name):
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        resp1 = gms(model_name,relative_entity_name)
        return resp1
    except(rospy.ServiceException, e):
        print("Service call failed: %s"%e)
        
def gls_client(model_name, link_name, relative_entity_name): #unused
    rospy.wait_for_service('/gazebo/get_link_state')
    gls = rospy.ServiceProxy('gazebo/get_link_state', gazebo_msgs.srv.GetLinkState)
    resp1 = gls(link_name=model_name+'::'+link_name, reference_frame=relative_entity_name)
    return(resp1)

def reset_now(pub1, pub2, pub3, pub4, pub5):
    pass
    

def set_joint_positions(): # formerly set_angle()
    os.system('rosservice call /gazebo/reset_simulation')
    rospy.init_node('motion_controller2', anonymous=False) # arbitrary name for this node
    #pub1 = rospy.Publisher('leg_l_controller/follow_joint_trajectory/goal', control_msgs.msg.FollowJointTrajectoryActionGoal, queue_size=10)
    pub1 = rospy.Publisher('/leg_l_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=0)
    pub2 = rospy.Publisher('/leg_r_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=0)
    pub3 = rospy.Publisher('/toes_l_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=0)
    pub4 = rospy.Publisher('/toes_r_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=0)
    pub5 = rospy.Publisher('/spine_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=0)
    #pub3 = rospy.Publisher('ball_l_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=10)
    #pub4 = rospy.Publisher('ball_r_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=10)
    #pub5 = rospy.Publisher('spine_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=10)
    #clock_listener = rospy.Subscriber('/clock', Float64)
    #print(clock_listener)
    #rospy.rostime.
    wait = 0 # must be >= 1
    leg_l_jt = trajectory_msgs.msg.JointTrajectory()
    leg_l_jt.header.stamp.set(wait, 0) # controls how long to get to starting position
    leg_l_jt.joint_names = ['hip_l_x', 'hip_l_y', 'hip_l_z', 'knee_l', 'ankle_l_y', 'ankle_l_x']
    leg_l_jt.header.frame_id = 'pelvis'
    
    leg_r_jt = trajectory_msgs.msg.JointTrajectory()
    leg_r_jt.header.stamp.set(wait, 0) # controls how long to get to starting position
    leg_r_jt.joint_names = ['hip_r_x', 'hip_r_y', 'hip_r_z', 'knee_r', 'ankle_r_y', 'ankle_r_x']
    leg_r_jt.header.frame_id = 'pelvis'
    
    ball_l_jt = trajectory_msgs.msg.JointTrajectory()
    ball_l_jt.header.stamp.set(wait, 0) # controls how long to get to starting position
    ball_l_jt.joint_names = ['ball_joint_l']
    ball_l_jt.header.frame_id = 'pelvis'
    
    ball_r_jt = trajectory_msgs.msg.JointTrajectory()
    ball_r_jt.header.stamp.set(wait, 0) # controls how long to get to starting position
    ball_r_jt.joint_names = ['ball_joint_r']
    ball_r_jt.header.frame_id = 'pelvis'
    
    spine_jt = trajectory_msgs.msg.JointTrajectory()
    spine_jt.header.stamp.set(wait, 0) # controls how long to get to starting position
    spine_jt.joint_names = ['spine_x', 'spine_y', 'spine_z']
    spine_jt.header.frame_id = 'pelvis'
    #print(leg_r_jt.header)
    
    #now = rospy.get_rostime()
    #while True:
    rospy.sleep(.001)
    now = rospy.Time.now()
    #now = rospy.get_time()
    print('secs:{}  nsecs:{}'.format(now.secs, now.nsecs))
    #nsecs_interval = 900000000 # controls how long it takes between points


    '''
    points.effort = [20, 20, 20, 20, 20, 20]
    points.velocities = [20, 20, 20, 20, 20, 20]
    points.accelerations = [1, 1, 1, 1, 1, 1]
    '''

    #print(type(leg_r_jt.joint_names))
    hz = 100
    rate = rospy.Rate(hz) # 100hz
    '''
    for _ in range(hz): # trajectory MUST use hz!
        
        pub2.publish(leg_r_jt.header, leg_r_jt.joint_names, leg_r_jt.points)
        rate.sleep()

    '''
    
    model_name = 'robot'
    relative_entity_name = 'world'
    # initialize
    i = 0
    k = 0
    v = False
    #vx = vy = vz = vax = vay = vaz = 1
    '''
    leg_l_pub = rospy.Publisher('/leg_l_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=10)
    leg_l_jt = trajectory_msgs.msg.JointTrajectory()
    leg_l_jt.header.stamp.set(1, 0)
    leg_l_jt.points = [1, 0, 0, 0, 0, 0]
    leg_l_jt.joint_names = ['hip_l_x', 'hip_l_y', 'hip_l_z', 'knee_l', 'ankle_l_y', 'ankle_l_x']
    leg_l_pub.publish(leg_l_jt)
    '''
    client = comm_test5.Client(55430)
    roll = pitch = yaw = 0.0
    #Jt = trajectory_msgs.msg.JointTrajectory()
    #Jt.header.stamp.set(1, 0)
    instant = True
    while not rospy.is_shutdown():
        print('===== i:{} ====='.format(i))
        if i > 0:
            msg = client.receive_message()
            #pause_simulation()
            #print('msg:{}'.format(msg))
            #try:
            #target_angles = [float(msg.split(' ')[0]), float(msg.split(' ')[1])]  # ex. '0.0 1.57' ==> [0.0, 1.57]
            target_angles = [[float(msg.split(' ')[0]), float(msg.split(' ')[1]), float(msg.split(' ')[2]), float(msg.split(' ')[3]), float(msg.split(' ')[4]), float(msg.split(' ')[5])], 
                              [float(msg.split(' ')[6]), float(msg.split(' ')[7]), float(msg.split(' ')[8]), float(msg.split(' ')[9]), float(msg.split(' ')[10]), float(msg.split(' ')[11])], 
                              [float(msg.split(' ')[12])], 
                              [float(msg.split(' ')[13])],
                              [float(msg.split(' ')[14]), float(msg.split(' ')[15]), float(msg.split(' ')[16])]]
            duration_float = float(msg.split(' ')[17])
            get_min_x = int(msg.split(' ')[18])
            get_min_y = int(msg.split(' ')[19])
            print('msg[18]:{} msg[19:{} get_min_x:{} get_min_y:{}'.format(msg[18], msg[19], get_min_x, get_min_y))
            print('duration_float:{}'.format(duration_float))
        else:
            #pause_simulation()
            target_angles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0], [0.0], [0.0, 0.0, 0.0]]
            starting_angles = target_angles
            duration_float = 1.0
            get_min_x = True
            get_min_y = True

        duration = (int(duration_float), (float(duration_float) - int(duration_float)) * 1000000000)

        
        leg_l_starts = [x for x in starting_angles[0]]
        leg_l_ends = [x for x in target_angles[0]]
        leg_r_starts = [x for x in starting_angles[1]]
        leg_r_ends = [x for x in target_angles[1]]
        ball_l_starts = [x for x in starting_angles[2]]
        ball_l_ends = [x for x in target_angles[2]]
        ball_r_starts = [x for x in starting_angles[3]]
        ball_r_ends = [x for x in target_angles[3]]
        spine_starts = [x for x in starting_angles[4]]
        spine_ends = [x for x in target_angles[4]]
        
        num_points = int(hz * duration_float)
        if instant == False:
            
    
            leg_l_joint_intervals = [(leg_l_ends[j] - leg_l_starts[j]) / (num_points-1) for j in range(len(leg_l_starts))]
            leg_r_joint_intervals = [(leg_r_ends[j] - leg_r_starts[j]) / (num_points-1) for j in range(len(leg_r_starts))]
            ball_l_joint_intervals = [(ball_l_ends[j] - ball_l_starts[j]) / (num_points-1) for j in range(len(ball_l_starts))]
            ball_r_joint_intervals = [(ball_r_ends[j] - ball_r_starts[j]) / (num_points-1) for j in range(len(ball_r_starts))]
            spine_joint_intervals = [(spine_ends[j] - spine_starts[j]) / (num_points-1) for j in range(len(spine_starts))]
        elif instant == True:
            #num_points = 1
            leg_l_joint_intervals = [leg_l_ends[j] - leg_l_starts[j] for j in range(len(leg_l_starts))]
            leg_r_joint_intervals = [leg_r_ends[j] - leg_r_starts[j] for j in range(len(leg_r_starts))]
            ball_l_joint_intervals = [ball_l_ends[j] - ball_l_starts[j] for j in range(len(ball_l_starts))]
            ball_r_joint_intervals = [ball_r_ends[j] - ball_r_starts[j] for j in range(len(ball_r_starts))]
            spine_joint_intervals = [spine_ends[j] - spine_starts[j] for j in range(len(spine_starts))]
            
        now = rospy.Time.now()
        if k == 0:
            sim_start_time = now.secs + now.nsecs/1000000000.
        leg_l_jt.header.stamp.set(now.secs, now.nsecs)
        leg_r_jt.header.stamp.set(now.secs, now.nsecs)
        ball_l_jt.header.stamp.set(now.secs, now.nsecs)
        ball_r_jt.header.stamp.set(now.secs, now.nsecs)
        spine_jt.header.stamp.set(now.secs, now.nsecs)
        '''
        points.effort = [20, 20, 20, 20, 20, 20]
        points.velocities = [20, 20, 20, 20, 20, 20]
        points.accelerations = [1, 1, 1, 1, 1, 1]
        '''
        now.secs = 0
        now.nsecs = 0
        if instant == False:
            leg_l_jt.points = [JTPoint(x, num_points, now, duration, leg_l_starts, leg_l_joint_intervals).point for x in range(num_points)] #[leg_l_point0.point, leg_l_point1.point, leg_l_point2.point, ...]
            leg_r_jt.points = [JTPoint(x, num_points, now, duration, leg_r_starts, leg_r_joint_intervals).point for x in range(num_points)]  #[leg_r_point0.point, leg_r_point1.point, leg_r_point2.point, ...]
            ball_l_jt.points = [JTPoint(x, num_points, now, duration, ball_l_starts, ball_l_joint_intervals).point for x in range(num_points)]
            ball_r_jt.points = [JTPoint(x, num_points, now, duration, ball_r_starts, ball_r_joint_intervals).point for x in range(num_points)]
            spine_jt.points = [JTPoint(x, num_points, now, duration, spine_starts, spine_joint_intervals).point for x in range(num_points)]
        elif instant == True:
            leg_l_jt.points = [JTPoint(1, 1, now, 0, leg_l_starts, leg_l_joint_intervals, instant=instant).point for x in range(1)]
            leg_r_jt.points = [JTPoint(1, 1, now, 0, leg_r_starts, leg_r_joint_intervals, instant=instant).point for x in range(1)]  #[leg_r_point0.point, leg_r_point1.point, leg_r_point2.point, ...]
            ball_l_jt.points = [JTPoint(1, 1, now, 0, ball_l_starts, ball_l_joint_intervals, instant=instant).point for x in range(1)]
            ball_r_jt.points = [JTPoint(1, 1, now, 0, ball_r_starts, ball_r_joint_intervals, instant=instant).point for x in range(1)]
            spine_jt.points = [JTPoint(1, 1, now, 0, spine_starts, spine_joint_intervals, instant=instant).point for x in range(1)]

        print('DURATION:{}'.format(duration))
        #unpause_simulation()
        pub1.publish(leg_l_jt.header, leg_l_jt.joint_names, leg_l_jt.points)
        pub2.publish(leg_r_jt.header, leg_r_jt.joint_names, leg_r_jt.points)
        pub3.publish(ball_l_jt.header, ball_l_jt.joint_names, ball_l_jt.points)
        pub4.publish(ball_r_jt.header, ball_r_jt.joint_names, ball_r_jt.points)
        pub5.publish(spine_jt.header, spine_jt.joint_names, spine_jt.points)
        
        broken = 0
        for _ in range(num_points):
            
            link_state = rospy.ServiceProxy('/gazebo/get_link_state', gazebo_msgs.srv.GetLinkState)
            lower_spine_coordinates = link_state('torso_lower', '')
            lower_spine_orientation = lower_spine_coordinates.link_state.pose.orientation
            rpy = euler_from_quaternion([lower_spine_orientation.w, lower_spine_orientation.x, lower_spine_orientation.y, lower_spine_orientation.z])
            roll = rpy[1]
            pitch = rpy[2]
            yaw = rpy[0]
            
            if abs(roll) >= .6 or pitch >= .6 or pitch <= -.6:
                broken = 1
                break
            if v == True:
                rospy.loginfo(leg_l_jt) # will probably cause duplicates
                rospy.loginfo(leg_r_jt)
                


            
            rate.sleep()
        #time.sleep(wait-1)
        '''
        if wait - 1 >= 0:
            rospy.sleep(wait-1)
        else:
            rospy.sleep(wait) #TODO: Maybe this is throwing things off?
        '''
        starting_angles = target_angles
        #res = gms_client(model_name,relative_entity_name)
        #vax = GetModelStateResponse.twist.angular.x
        #p = gazebo_msgs.srv.GetLinkPropertiesResponse.
        
        #x_pos = res.pose.position.x # give penalty for moving in x direction
        #y_pos = res.pose.position.y
        #z_pos = res.pose.position.z # give penalty and restart sim when z falls too low
        
        model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
        object_coordinates = model_coordinates("robot", "")
        
        #link_coordinates = link_state('torso_upper', '')
        #spine_z = link_coordinates.link_state.pose.position.z
        #z_position = object_coordinates.pose.position.z
        #y_position = object_coordinates.pose.position.y
        #x_position = object_coordinates.pose.position.x
        
        #quaternion = (res.pose.orientation.x, res.pose.orientation.y, res.pose.orientation.z, res.pose.orientation.w)
        #euler = tf.transformations.euler_from_quaternion(quaternion)
        #roll = euler[0]
        #pitch = euler[1]
        #yaw = euler[2]
        if v == True:
            print('sending1, time:{}'.format(time.time()))
        link_state = rospy.ServiceProxy('/gazebo/get_link_state', gazebo_msgs.srv.GetLinkState)
        torso_lower_coordinates = link_state('torso_lower', '') # for some reason, pelvis returns model state
        torso_upper_coordinates = link_state('torso_upper', '')
        foot_l_coordinates = link_state('foot_l', '')
        foot_r_coordinates = link_state('foot_r', '')
        calf_l_coordinates = link_state('calf_l', '')
        calf_r_coordinates = link_state('calf_r', '')
        torso_lower_y = torso_lower_coordinates.link_state.pose.position.y
        torso_upper_y = torso_upper_coordinates.link_state.pose.position.y
        foot_l_y = foot_l_coordinates.link_state.pose.position.y
        foot_r_y = foot_r_coordinates.link_state.pose.position.y
        torso_lower_x = torso_lower_coordinates.link_state.pose.position.x
        torso_upper_x = torso_upper_coordinates.link_state.pose.position.x
        foot_l_x = foot_l_coordinates.link_state.pose.position.x
        foot_r_x = foot_r_coordinates.link_state.pose.position.x
        #calf_l_y = calf_l_coordinates.link_state.pose.position.y
        #calf_r_y = calf_r_coordinates.link_state.pose.position.y
        use_avg_foot_location = True
        # Get position of feet
        if use_avg_foot_location == True:
            adj_x = (foot_l_x + foot_r_x) / 2
            adj_y = (foot_l_y + foot_r_y) / 2
        else:
            adj_x = min(torso_lower_x, foot_l_x, foot_r_x) if get_min_x == True else max(torso_lower_x, foot_l_x, foot_r_x)
            adj_y = min(torso_lower_y, foot_l_y, foot_r_y) if get_min_y == True else max(torso_lower_y, foot_l_y, foot_r_y)


        print('get_min_x:{} get_min_y:{} adj_x:{} adj_y:{}'.format(get_min_x, get_min_y, adj_x, adj_y))
        print('min_x:{} max_x:{}'.format(min(torso_lower_x, torso_upper_x, foot_l_x, foot_r_x), max(torso_lower_x, torso_upper_x, foot_l_x, foot_r_x)))
        

        client.send_message('rpy:{}:{}:{} dist:{}:{}'.format('dummy', 'dummy', 'dummy', adj_x, adj_y)) # may want to change to y_position
        buffer1 = client.receive_message()
        #received
        #if v == True:
        
        link_state = rospy.ServiceProxy('/gazebo/get_link_state', gazebo_msgs.srv.GetLinkState)
        lower_spine_coordinates = link_state('torso_lower', '')
        upper_spine_coordinates = link_state('torso_upper', '')

        #upper_spine_z = upper_spine_coordinates.link_state.pose.position.z
        lower_spine_orientation = lower_spine_coordinates.link_state.pose.orientation

        
        
        #link_state = rospy.ServiceProxy('/gazebo/get_link_state', gazebo_msgs.srv.GetLinkState)
        #upper_spine_coordinates = link_state('torso_upper', '')
        lower_spine_twist = lower_spine_coordinates.link_state.twist
        rpy_vel_x = lower_spine_twist.angular.x
        rpy_vel_y = lower_spine_twist.angular.y
        rpy_vel_z = lower_spine_twist.angular.z
        
        vx = lower_spine_twist.linear.x
        vy = lower_spine_twist.linear.y
        vz = lower_spine_twist.linear.z
        
        lower_spine_pose = lower_spine_coordinates.link_state.pose
        position_x_spine = lower_spine_pose.position.x
        position_y_spine = lower_spine_pose.position.y
        position_z_spine = lower_spine_pose.position.z
        
        
        rpy = euler_from_quaternion([lower_spine_orientation.w, lower_spine_orientation.x, lower_spine_orientation.y, lower_spine_orientation.z])
        roll = rpy[1]
        pitch = rpy[2]
        yaw = rpy[0]
        
        
        leg_l_actual_joint_states = rospy.wait_for_message('/leg_l_controller/state', control_msgs.msg.JointTrajectoryControllerState).actual.positions
        leg_r_actual_joint_states = rospy.wait_for_message('/leg_r_controller/state', control_msgs.msg.JointTrajectoryControllerState).actual.positions
        hip_l_x, hip_l_y, hip_l_z, knee_l, ankle_l_x, ankle_l_y = leg_l_actual_joint_states
        hip_r_x, hip_r_y, hip_r_z, knee_r, ankle_r_x, ankle_r_y = leg_r_actual_joint_states
        sim_time = rospy.Time.now()
        print(sim_time)
        sim_time = sim_time.secs + sim_time.nsecs/1000000000. - sim_start_time
        print(sim_time)
        print(sim_start_time)
        
        #print('Z:{}'.format(upper_spine_z))
        #if upper_spine_z < 0.8:
        
        
        
        fallen_status = 0
            
        k += 1
        #if abs(roll) >= .6 or pitch >= .6 or pitch <= -.6 or sim_time >= 25.0 or broken == 1:
        if abs(roll) >= .6 or pitch >= .6 or pitch <= -.6 or broken == 1:
            k = 0

            target_angles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0], [0.0], [0.0, 0.0, 0.0]]
            #starting_angles = target_angles
            
            duration_float = .10
            duration = (int(duration_float), (float(duration_float) - int(duration_float)) * 1000000000)
            num_points = int(hz * duration_float)
            
            fallen_status = 1 # should be terminal status; doesn't imply losing
            print('RESETTING...\n')
            # reset joint positions to 0
            
            leg_l_starts = [x for x in starting_angles[0]]
            leg_l_ends = [x for x in target_angles[0]]
            leg_r_starts = [x for x in starting_angles[1]]
            leg_r_ends = [x for x in target_angles[1]]
            ball_l_starts = [x for x in starting_angles[2]]
            ball_l_ends = [x for x in target_angles[2]]
            ball_r_starts = [x for x in starting_angles[3]]
            ball_r_ends = [x for x in target_angles[3]]
            spine_starts = [x for x in starting_angles[4]]
            spine_ends = [x for x in target_angles[4]]
            
            #pause_simulation()
            
            now = rospy.Time.now()
            
            leg_l_jt.header.stamp.set(now.secs, now.nsecs)
            leg_r_jt.header.stamp.set(now.secs, now.nsecs)
            ball_l_jt.header.stamp.set(now.secs, now.nsecs)
            ball_r_jt.header.stamp.set(now.secs, now.nsecs)
            spine_jt.header.stamp.set(now.secs, now.nsecs)
            now.secs = 0
            now.nsecs = 0
            #now = (0, 0)
            
            leg_l_joint_intervals = [(leg_l_ends[j] - leg_l_starts[j]) / (num_points-1) for j in range(len(leg_l_starts))]
            leg_r_joint_intervals = [(leg_r_ends[j] - leg_r_starts[j]) / (num_points-1) for j in range(len(leg_r_starts))]
            ball_l_joint_intervals = [(ball_l_ends[j] - ball_l_starts[j]) / (num_points-1) for j in range(len(ball_l_starts))]
            ball_r_joint_intervals = [(ball_r_ends[j] - ball_r_starts[j]) / (num_points-1) for j in range(len(ball_r_starts))]
            spine_joint_intervals = [(spine_ends[j] - spine_starts[j]) / (num_points-1) for j in range(len(spine_starts))]
            


            leg_l_jt.points = [JTPoint(x, num_points, now, duration, leg_l_starts, leg_l_joint_intervals).point for x in range(num_points)]
            leg_r_jt.points = [JTPoint(x, num_points, now, duration, leg_r_starts, leg_r_joint_intervals).point for x in range(num_points)]
            ball_l_jt.points = [JTPoint(x, num_points, now, duration, ball_l_starts, ball_l_joint_intervals).point for x in range(num_points)]
            ball_r_jt.points = [JTPoint(x, num_points, now, duration, ball_r_starts, ball_r_joint_intervals).point for x in range(num_points)]
            spine_jt.points = [JTPoint(x, num_points, now, duration, spine_starts, spine_joint_intervals).point for x in range(num_points)]
            
            #unpause_simulation()
            
            print('publishing...')
            pub1.publish(leg_l_jt.header, leg_l_jt.joint_names, leg_l_jt.points)
            pub2.publish(leg_r_jt.header, leg_r_jt.joint_names, leg_r_jt.points)
            pub3.publish(ball_l_jt.header, ball_l_jt.joint_names, ball_l_jt.points)
            pub4.publish(ball_r_jt.header, ball_r_jt.joint_names, ball_r_jt.points)
            pub5.publish(spine_jt.header, spine_jt.joint_names, spine_jt.points)
            
            
            for _ in range(num_points):

                rate.sleep()
            #gazebo_msgs.msg.ODEPhysics.sor_pgs_iters
            #gazebo_msgs.srv.SetPhysicsPropertiesRequest.ode_config.sor_pgs_iters
            print('done')
            
            starting_angles = target_angles
            #pause_simulation()
            os.system('rosservice call /gazebo/reset_simulation')
            #time.sleep(.05)
            #unpause_simulation()
        fallen = '{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}'.format(fallen_status,
                  roll, pitch, yaw, rpy_vel_x, rpy_vel_y, rpy_vel_z, position_x_spine, position_y_spine, position_z_spine, vx, vy, vz, sim_time,
                  hip_l_z, hip_l_x, hip_l_y, knee_l, ankle_l_y, ankle_l_x, hip_r_z, hip_r_x, hip_r_y, knee_r, ankle_r_y, ankle_r_x, adj_x, adj_y)
        print('buffer1 received')
        client.send_message(fallen)
        
        
        if i % 1000 == 0 and i != 0:
            print('receiving save signal')
            msg = client.receive_message()
            if msg == 'pause':
                pause_simulation()
            print('receiving resume signal')
            msg = client.receive_message()
            if msg == 'unpause':
                unpause_simulation()
        
        
        if v == True:
            print('fallen status:{}'.format(fallen))
        i += 1
            
        #if i == 0:
            #time.sleep(1)
        '''
        except KeyboardInterrupt:
            break
        '''
if __name__ == '__main__':
    try:
        #TODO: send initial state
        #client.send_message()
        set_joint_positions()
    except rospy.ROSInterruptException:
        pass
	