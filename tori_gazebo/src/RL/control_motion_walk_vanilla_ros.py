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
from tori_msgs.msg import State, Replay, ToriJointAngles

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
            print('Divide by zero; Setting interval to 0')
            interval = 0.016 # use a small number to prevent ROS from dropping point
        self.point = trajectory_msgs.msg.JointTrajectoryPoint()
        self.point.positions = []
        for i in range(len(starts)):
            self.point.positions.append(starts[i] + joint_intervals[i]*point_number)
        current_point_time = now.secs + now.nsecs/1000000000. + interval*point_number
        secs, nsecs = int(current_point_time), (float(current_point_time) - int(current_point_time)) * 1000000000
        self.point.time_from_start.set(secs, nsecs)
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


computer = 'kelsey'
class ActionNode():
    def __init__(self):
        rospy.init_node('tori_joint_commander_{}'.format(computer), anonymous=False)
        self.action_sub = rospy.Subscriber('/tori_joint_command_{}'.format(computer), ToriJointAngles, self.action_callback)
        self.pub1 = rospy.Publisher('/leg_l_controller_{}/command'.format(computer), trajectory_msgs.msg.JointTrajectory, queue_size=0)
        self.pub2 = rospy.Publisher('/leg_r_controller_{}/command'.format(computer), trajectory_msgs.msg.JointTrajectory, queue_size=0)
        self.pub3 = rospy.Publisher('/toes_l_controller_{}/command'.format(computer), trajectory_msgs.msg.JointTrajectory, queue_size=0)
        self.pub4 = rospy.Publisher('/toes_r_controller_{}/command'.format(computer), trajectory_msgs.msg.JointTrajectory, queue_size=0)
        self.pub5 = rospy.Publisher('/spine_controller_{}/command'.format(computer), trajectory_msgs.msg.JointTrajectory, queue_size=0)
        self.state_pub = rospy.Publisher('/tori_state_{}'.format(computer), State, queue_size=0)
        
        
        wait = 0 # must be >= 1
        self.leg_l_jt = trajectory_msgs.msg.JointTrajectory()
        self.leg_l_jt.header.stamp.set(wait, 0) # controls how long to get to starting position?
        self.leg_l_jt.joint_names = ['hip_l_x', 'hip_l_y', 'hip_l_z', 'knee_l', 'ankle_l_y', 'ankle_l_x']
        self.leg_l_jt.header.frame_id = 'pelvis'
        
        self.leg_r_jt = trajectory_msgs.msg.JointTrajectory()
        self.leg_r_jt.header.stamp.set(wait, 0) 
        self.leg_r_jt.joint_names = ['hip_r_x', 'hip_r_y', 'hip_r_z', 'knee_r', 'ankle_r_y', 'ankle_r_x']
        self.leg_r_jt.header.frame_id = 'pelvis'
        
        self.ball_l_jt = trajectory_msgs.msg.JointTrajectory()
        self.ball_l_jt.header.stamp.set(wait, 0) 
        self.ball_l_jt.joint_names = ['ball_joint_l']
        self.ball_l_jt.header.frame_id = 'pelvis'
        
        self.ball_r_jt = trajectory_msgs.msg.JointTrajectory()
        self.ball_r_jt.header.stamp.set(wait, 0) 
        self.ball_r_jt.joint_names = ['ball_joint_r']
        self.ball_r_jt.header.frame_id = 'pelvis'
        
        self.spine_jt = trajectory_msgs.msg.JointTrajectory()
        self.spine_jt.header.stamp.set(wait, 0) 
        self.spine_jt.joint_names = ['spine_x', 'spine_y', 'spine_z']
        self.spine_jt.header.frame_id = 'pelvis'
        
        rospy.sleep(.001)
        now = rospy.Time.now()
        print('secs:{}  nsecs:{}'.format(now.secs, now.nsecs))
    
    
        self.hz = 100
        self.rate = rospy.Rate(self.hz) # 100hz
    
        self.model_name = 'robot'
        self.relative_entity_name = 'world'
        self.i = 0
        self.k = 0
    
        self.roll = self.pitch = self.yaw = 0.0
    
        self.instant = True
        

    def action_callback(self, msg):
        print('===== i:{} ====='.format(self.i))
        if self.i > 0:
            #pause_simulation()
            target_angles = msg.angles

            duration_float = msg.duration
        else:
            #pause_simulation()
            target_angles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0], [0.0], [0.0, 0.0, 0.0]]
            starting_angles = target_angles
            duration_float = 1.0
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
        
        num_points = int(self.hz * duration_float)
        if self.instant == False:
            
    
            leg_l_joint_intervals = [(leg_l_ends[j] - leg_l_starts[j]) / (num_points-1) for j in range(len(leg_l_starts))]
            leg_r_joint_intervals = [(leg_r_ends[j] - leg_r_starts[j]) / (num_points-1) for j in range(len(leg_r_starts))]
            ball_l_joint_intervals = [(ball_l_ends[j] - ball_l_starts[j]) / (num_points-1) for j in range(len(ball_l_starts))]
            ball_r_joint_intervals = [(ball_r_ends[j] - ball_r_starts[j]) / (num_points-1) for j in range(len(ball_r_starts))]
            spine_joint_intervals = [(spine_ends[j] - spine_starts[j]) / (num_points-1) for j in range(len(spine_starts))]
        elif self.instant == True:
            leg_l_joint_intervals = [leg_l_ends[j] - leg_l_starts[j] for j in range(len(leg_l_starts))]
            leg_r_joint_intervals = [leg_r_ends[j] - leg_r_starts[j] for j in range(len(leg_r_starts))]
            ball_l_joint_intervals = [ball_l_ends[j] - ball_l_starts[j] for j in range(len(ball_l_starts))]
            ball_r_joint_intervals = [ball_r_ends[j] - ball_r_starts[j] for j in range(len(ball_r_starts))]
            spine_joint_intervals = [spine_ends[j] - spine_starts[j] for j in range(len(spine_starts))]
            
        now = rospy.Time.now()
        if self.k == 0:
            sim_start_time = now.secs + now.nsecs/1000000000.
        self.leg_l_jt.header.stamp.set(now.secs, now.nsecs)
        self.leg_r_jt.header.stamp.set(now.secs, now.nsecs)
        self.ball_l_jt.header.stamp.set(now.secs, now.nsecs)
        self.ball_r_jt.header.stamp.set(now.secs, now.nsecs)
        self.spine_jt.header.stamp.set(now.secs, now.nsecs)

        now.secs = 0
        now.nsecs = 0
        if self.instant == False:
            self.leg_l_jt.points = [JTPoint(x, num_points, now, duration, leg_l_starts, leg_l_joint_intervals).point for x in range(num_points)] #[leg_l_point0.point, leg_l_point1.point, leg_l_point2.point, ...]
            self.leg_r_jt.points = [JTPoint(x, num_points, now, duration, leg_r_starts, leg_r_joint_intervals).point for x in range(num_points)]  #[leg_r_point0.point, leg_r_point1.point, leg_r_point2.point, ...]
            self.ball_l_jt.points = [JTPoint(x, num_points, now, duration, ball_l_starts, ball_l_joint_intervals).point for x in range(num_points)]
            self.ball_r_jt.points = [JTPoint(x, num_points, now, duration, ball_r_starts, ball_r_joint_intervals).point for x in range(num_points)]
            self.spine_jt.points = [JTPoint(x, num_points, now, duration, spine_starts, spine_joint_intervals).point for x in range(num_points)]
        elif self.instant == True:
            self.leg_l_jt.points = [JTPoint(1, 1, now, 0, leg_l_starts, leg_l_joint_intervals, instant=self.instant).point for x in range(1)]
            self.leg_r_jt.points = [JTPoint(1, 1, now, 0, leg_r_starts, leg_r_joint_intervals, instant=self.instant).point for x in range(1)]  #[leg_r_point0.point, leg_r_point1.point, leg_r_point2.point, ...]
            self.ball_l_jt.points = [JTPoint(1, 1, now, 0, ball_l_starts, ball_l_joint_intervals, instant=self.instant).point for x in range(1)]
            self.ball_r_jt.points = [JTPoint(1, 1, now, 0, ball_r_starts, ball_r_joint_intervals, instant=self.instant).point for x in range(1)]
            self.spine_jt.points = [JTPoint(1, 1, now, 0, spine_starts, spine_joint_intervals, instant=self.instant).point for x in range(1)]

        print('DURATION:{}'.format(duration))
        #unpause_simulation()
        self.pub1.publish(self.leg_l_jt.header, self.leg_l_jt.joint_names, self.leg_l_jt.points)
        self.pub2.publish(self.leg_r_jt.header, self.leg_r_jt.joint_names, self.leg_r_jt.points)
        self.pub3.publish(self.ball_l_jt.header, self.ball_l_jt.joint_names, self.ball_l_jt.points)
        self.pub4.publish(self.ball_r_jt.header, self.ball_r_jt.joint_names, self.ball_r_jt.points)
        self.pub5.publish(self.spine_jt.header, self.spine_jt.joint_names, self.spine_jt.points)
        
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


            self.rate.sleep()

        starting_angles = target_angles
        
        model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
        object_coordinates = model_coordinates("robot", "")
        
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
        calf_l_y = calf_l_coordinates.link_state.pose.position.y
        calf_r_y = calf_r_coordinates.link_state.pose.position.y
        min_distance = min(torso_lower_y, torso_upper_y, foot_l_y, foot_r_y, calf_l_y, calf_r_y)

        link_state = rospy.ServiceProxy('/gazebo/get_link_state', gazebo_msgs.srv.GetLinkState)
        lower_spine_coordinates = link_state('torso_lower', '')
        upper_spine_coordinates = link_state('torso_upper', '')

        lower_spine_orientation = lower_spine_coordinates.link_state.pose.orientation

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
            
        self.k += 1
        if abs(roll) >= .6 or pitch >= .6 or pitch <= -.6 or sim_time >= 25.0 or broken == 1:
            k = 0

            target_angles = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0], [0.0], [0.0, 0.0, 0.0]]
            
            duration_float = .10
            duration = (int(duration_float), (float(duration_float) - int(duration_float)) * 1000000000)
            num_points = int(self.hz * duration_float)
            
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
            
            self.leg_l_jt.header.stamp.set(now.secs, now.nsecs)
            self.leg_r_jt.header.stamp.set(now.secs, now.nsecs)
            self.ball_l_jt.header.stamp.set(now.secs, now.nsecs)
            self.ball_r_jt.header.stamp.set(now.secs, now.nsecs)
            self.spine_jt.header.stamp.set(now.secs, now.nsecs)
            now.secs = 0
            now.nsecs = 0
            #now = (0, 0)
            
            leg_l_joint_intervals = [(leg_l_ends[j] - leg_l_starts[j]) / (num_points-1) for j in range(len(leg_l_starts))]
            leg_r_joint_intervals = [(leg_r_ends[j] - leg_r_starts[j]) / (num_points-1) for j in range(len(leg_r_starts))]
            ball_l_joint_intervals = [(ball_l_ends[j] - ball_l_starts[j]) / (num_points-1) for j in range(len(ball_l_starts))]
            ball_r_joint_intervals = [(ball_r_ends[j] - ball_r_starts[j]) / (num_points-1) for j in range(len(ball_r_starts))]
            spine_joint_intervals = [(spine_ends[j] - spine_starts[j]) / (num_points-1) for j in range(len(spine_starts))]
            


            self.leg_l_jt.points = [JTPoint(x, num_points, now, duration, leg_l_starts, leg_l_joint_intervals).point for x in range(num_points)]
            self.leg_r_jt.points = [JTPoint(x, num_points, now, duration, leg_r_starts, leg_r_joint_intervals).point for x in range(num_points)]
            self.ball_l_jt.points = [JTPoint(x, num_points, now, duration, ball_l_starts, ball_l_joint_intervals).point for x in range(num_points)]
            self.ball_r_jt.points = [JTPoint(x, num_points, now, duration, ball_r_starts, ball_r_joint_intervals).point for x in range(num_points)]
            self.spine_jt.points = [JTPoint(x, num_points, now, duration, spine_starts, spine_joint_intervals).point for x in range(num_points)]
            
            #unpause_simulation()
            
            print('publishing...')
            # These control the actual joints on the robot
            self.pub1.publish(self.leg_l_jt.header, self.leg_l_jt.joint_names, self.leg_l_jt.points)
            self.pub2.publish(self.leg_r_jt.header, self.leg_r_jt.joint_names, self.leg_r_jt.points)
            self.pub3.publish(self.ball_l_jt.header, self.ball_l_jt.joint_names, self.ball_l_jt.points)
            self.pub4.publish(self.ball_r_jt.header, self.ball_r_jt.joint_names, self.ball_r_jt.points)
            self.pub5.publish(self.spine_jt.header, self.spine_jt.joint_names, self.spine_jt.points)
            
            
            for _ in range(num_points):

                self.rate.sleep()
            print('done')
            
            starting_angles = target_angles
            #pause_simulation()
            os.system('rosservice call /gazebo/reset_simulation')
            #unpause_simulation()
        fallen = '{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}'.format(fallen_status,
                  roll, pitch, yaw, rpy_vel_x, rpy_vel_y, rpy_vel_z, position_x_spine, position_z_spine, vx, vy, vz, sim_time,
                  hip_l_z, hip_l_x, hip_l_y, knee_l, ankle_l_y, ankle_l_x, hip_r_z, hip_r_x, hip_r_y, knee_r, ankle_r_y, ankle_r_x)
        print('buffer1 received')
        state = State()
        state.fallen_status = fallen_status
        state.orientation = [roll, pitch, yaw]
        state.pos = [position_x_spine, position_y_spine, position_z_spine] #TODO: get position_y_spine, not necessarily minimin
        state.distance_minimum = min_distance
        state.rpy_vel = [rpy_vel_x, rpy_vel_y, rpy_vel_z]
        state.vel = [vx, vy, vz]
        state.sim_time = sim_time
        self.state_pub.publish(state)
        
        
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
            

if __name__ == '__main__':
    try:
        action_node = ActionNode()
        action_sub = action_node.action_sub
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
	