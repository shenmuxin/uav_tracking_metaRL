#! /usr/bin/env python
# coding :utf-8

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Quaternion, TwistStamped, Accel
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget, ExtendedState
from mavros_msgs.srv import SetMode, CommandBool

import tf2_ros
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from pymavlink import mavutil
from threading import Thread
from six.moves import xrange
import math
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


PX4_CTRL_DICT = {
    "pos":{
        "mavros_pos_ctl": True,
        "mavros_vel_ctl": False,
        "mavros_acc_ctl": False,
    },
    "vel":{
        "mavros_pos_ctl": False,
        "mavros_vel_ctl": True,
        "mavros_acc_ctl": False,
    },
    # "acc":{
    #     "mavros_pos_ctl": False,
    #     "mavros_vel_ctl": False,
    #     "mavros_acc_ctl": True,
    # }
}

class PX4OffboardController:

    def __init__(self, robot_name='iris', offset=[7,0], debug=False):
        
        # trajectory record
        self.trajectory_record_flag = False
        # specify robot_name, default is iris
        self.robot_name = robot_name

        # specify the original coordinate
        self.offset = offset
        # if debug or not
        self.debug = debug

        # initialize callback msg store space
        self.state = State()
        self.local_position = PoseStamped()
        self.local_velocity = TwistStamped()
        self.extended_state = ExtendedState()

        # initialize publish msg store space
        self.command_pos = self.construct_pos_target()
        self.command_vel = self.construct_vel_target()
        # self.command_acc = self.construct_acc_target()

        # flag for topics ready
        self.sub_topics_ready = {
            key: False
            for key in [
                'local_pos',
                'local_vel',
                'state',
                'ext_state'
            ]
        }

        # ROS services
        service_timeout = 30
        if self.debug:
            rospy.loginfo("waiting for ROS services")
        try:
            rospy.wait_for_service(self.robot_name + '/mavros/cmd/arming', service_timeout)
            rospy.wait_for_service(self.robot_name + '/mavros/set_mode', service_timeout)
            if self.debug:
                rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            rospy.loginfo("failed to connect to services")

        self.set_arming_srv = rospy.ServiceProxy(self.robot_name + '/mavros/cmd/arming',CommandBool)
        self.set_mode_srv = rospy.ServiceProxy(self.robot_name + '/mavros/set_mode', SetMode)


        # ROS subscribers        
        self.local_pos_sub = rospy.Subscriber(self.robot_name + '/mavros/local_position/pose', PoseStamped, self.local_position_CB)
        self.local_vel_sub = rospy.Subscriber(self.robot_name + '/mavros/local_position/velocity_body', TwistStamped, self.local_velocity_CB)
        self.state_sub = rospy.Subscriber(self.robot_name + '/mavros/state', State, self.state_CB)
        self.ext_state_sub = rospy.Subscriber(self.robot_name + '/mavros/extended_state', ExtendedState, self.extended_state_CB)


        # ROS publishers
        self.set_posTarget_pub =  rospy.Publisher(self.robot_name + '/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        # self.set_accTarget_pub =  rospy.Publisher(self.robot_name + '/mavros/setpoint_accel/accel', Accel, queue_size=10)
        

        # trajectory data store in [x, y, z] form in world frame
        self.trajectory_data = np.array([self.offset[0], self.offset[1], 0]).reshape(-1,3)
        

        # control mode flag
        self.controller_switch('pos') 

        # begin threads
        self.Thread_start()

    

    #================================== Thread Loop Function Begin ==================================

    def Thread_start(self):
        # mavros thread
        mavros_pub_thread = Thread(target=self.mavros_pub_loop, args=())
        mavros_pub_thread.daemon = True   # kill the thread if exit
        mavros_pub_thread.start()

    def mavros_pub_loop(self):
        rate = rospy.Rate(10)  # Hz
        while (not rospy.is_shutdown()):
            if self.mavros_pos_ctl:
                self.set_posTarget_pub.publish(self.command_pos)
            if self.mavros_vel_ctl:
                self.set_posTarget_pub.publish(self.command_vel)
            if self.mavros_acc_ctl:
                self.set_posTarget_pub.publish(self.command_acc)

            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
    

    #================================== Thread Loop Function End ==================================


    #================================== Callback Functions Begin ==================================
    def state_CB(self, msg):
        if self.state.armed != msg.armed:
            if self.debug:
                rospy.loginfo(self.robot_name + " armed state changed from {0} to {1}".format(
                    self.state.armed, msg.armed))

        if self.state.connected != msg.connected:
            if self.debug:
                rospy.loginfo(self.robot_name + " connected changed from {0} to {1}".format(
                    self.state.connected, msg.connected))

        if self.state.mode != msg.mode:
            if self.debug:
                rospy.loginfo(self.robot_name + " mode changed from {0} to {1}".format(
                    self.state.mode, msg.mode))

        if self.state.system_status != msg.system_status:
            if self.debug:
                rospy.loginfo(self.robot_name + " system_status changed from {0} to {1}".format(
                    mavutil.mavlink.enums['MAV_STATE'][self.state.system_status].name, 
                    mavutil.mavlink.enums['MAV_STATE'][msg.system_status].name))

        # update state
        self.state = msg

        # mavros publishes a disconnected state message on init
        if not self.sub_topics_ready['state'] and msg.connected:
            self.sub_topics_ready['state'] = True


    def local_position_CB(self, msg):
        self.local_position.pose = msg.pose

        if not self.sub_topics_ready['local_pos']:
            self.sub_topics_ready['local_pos'] = True
        
        if self.trajectory_record_flag:
            self.record_traject(msg.pose.position.x + self.offset[0], 
                                msg.pose.position.y + self.offset[1], 
                                msg.pose.position.z, 0.01)


    def local_velocity_CB(self, msg):
        self.local_velocity = msg

        if not self.sub_topics_ready['local_vel']:
            self.sub_topics_ready['local_vel'] = True
            
    
    def extended_state_CB(self, data):
        if self.extended_state.vtol_state != data.vtol_state:
            if self.debug:
                rospy.loginfo(self.robot_name + " VTOL state changed from {0} to {1}".format(
                    mavutil.mavlink.enums['MAV_VTOL_STATE']
                    [self.extended_state.vtol_state].name, mavutil.mavlink.enums[
                        'MAV_VTOL_STATE'][data.vtol_state].name))

        if self.extended_state.landed_state != data.landed_state:
            if self.debug:
                rospy.loginfo(self.robot_name + " landed state changed from {0} to {1}".format(
                    mavutil.mavlink.enums['MAV_LANDED_STATE']
                    [self.extended_state.landed_state].name, mavutil.mavlink.enums[
                        'MAV_LANDED_STATE'][data.landed_state].name))

        self.extended_state = data

        if not self.sub_topics_ready['ext_state']:
            self.sub_topics_ready['ext_state'] = True

    #================================== Callback Functions End ==================================


    #================================== Trajectory Tool Functions Begin ==================================
    def set_record(self, flag):
        """
        func->
            begin trajectory record or not
        para->
            flag: True if begin
        """
        if self.debug:
            rospy.loginfo(self.robot_name + " set trajectory record flag from {0} to {1}".format(self.trajectory_record_flag, flag))
        self.trajectory_record_flag = flag
    
    def record_traject(self, x, y, z, offset):
        """
        func->
            record trajectory if the distance between two points greater than offset
        """
        current = np.array((x, y, z))
        position = np.array((self.trajectory_data[:,0][-1],
                        self.trajectory_data[:,1][-1],
                        self.trajectory_data[:,2][-1]))
        if np.linalg.norm(current - position) >= offset:
            self.trajectory_data = np.concatenate((self.trajectory_data, np.array([[x, y, z]])), axis=0)
    
    def get_trajectory(self):
        return self.trajectory_data

    def clear_traject(self):
        """
        func->
            clear trajectory data
        """
        if self.debug:
            rospy.loginfo(self.robot_name + " clear trajectory data")
        self.trajectory_data = np.array([self.offset[0], self.offset[1], 0]).reshape(-1,3)

    def plot_traject3d(self):
        """
        func->
            plot trajectory curve in 3D
        """
        if len(self.trajectory_data) <= 2:
            rospy.loginfo(self.robot_name + " No trajectory data to plot")
        rospy.loginfo(self.robot_name + " record {0} point of trajectory".format(len(self.trajectory_data)))

        x = self.trajectory_data[:,0]
        y = self.trajectory_data[:,1]
        z = self.trajectory_data[:,2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='uav trajectory')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title('Trajectory in 3D')

        # ax.set_xlim(0, 10)
        # ax.set_ylim(0, 10)
        # ax.set_zlim(-10, 10)
        plt.show()
    
    def plot_traject2d(self):
        """
        func->
            plot trajectory curve in 2D
        """
        if len(self.trajectory_data) <= 2:
            rospy.loginfo(self.robot_name + " No trajectory data to plot")
        rospy.loginfo(self.robot_name + " record {0} point of trajectory".format(len(self.trajectory_data)))

        x = self.trajectory_data[:,0]
        y = self.trajectory_data[:,1]
        z = self.trajectory_data[:,2]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y, label='uav trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Trajectory in 2D')
        plt.show()

    #================================== Trajectory Tool Functions End ==================================


    #================================== Helper Methods Begin ==================================
        
    def construct_pos_target(self, x=0, y=0, z=0, yaw=0, frame="ENU"):
        '''
        func->
            construct position msg
        param->
            frame: FLU means body-frame forward-left-up
                    ENU means world-frame east-north-up
        remark->
            uint8 coordinate_frame
            uint8 FRAME_LOCAL_NED = 1
            uint8 FRAME_LOCAL_OFFSET_NED = 7
            uint8 FRAME_BODY_NED = 8
            uint8 FRAME_BODY_OFFSET_NED = 9
        '''
        
        coordinate_frame = 1
        if frame == "FLU": 
            coordinate_frame = 8
        elif frame == "ENU": 
            coordinate_frame = 1

        target_raw_pose = PositionTarget()
        target_raw_pose.header.stamp = rospy.Time.now()
        target_raw_pose.coordinate_frame = coordinate_frame

        target_raw_pose.type_mask = (
            PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ 
            + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ 
            + PositionTarget.FORCE + PositionTarget.IGNORE_YAW_RATE
        )
        target_raw_pose.position.x = x
        target_raw_pose.position.y = y
        target_raw_pose.position.z = z

        target_raw_pose.yaw = yaw
        return target_raw_pose 


    def construct_vel_target(self, vx=0, vy=0, vz=0, yaw_rate=0,frame="ENU"):   
        '''
        func->
            construct velocity msg
        param->
            frame: FLU means body-frame forward-left-up
                    ENU means world-frame east-north-up
        remarks->
            uint8 coordinate_frame
            uint8 FRAME_LOCAL_NED = 1
            uint8 FRAME_LOCAL_OFFSET_NED = 7
            uint8 FRAME_BODY_NED = 8
            uint8 FRAME_BODY_OFFSET_NED = 9
        '''
        
        coordinate_frame = 1
        if frame == "FLU": 
            coordinate_frame = 8
        elif frame == "ENU": 
            coordinate_frame = 1

        target_raw_pose = PositionTarget()
        target_raw_pose.header.stamp = rospy.Time.now()

        target_raw_pose.coordinate_frame = coordinate_frame 
        target_raw_pose.type_mask = (
            PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ 
            + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + PositionTarget.IGNORE_AFZ 
            + PositionTarget.FORCE + PositionTarget.IGNORE_YAW 
        )
        target_raw_pose.velocity.x = vx
        target_raw_pose.velocity.y = vy
        target_raw_pose.velocity.z = vz

        target_raw_pose.yaw_rate = yaw_rate
        return target_raw_pose

    
    # def construct_acc_target(self, ax=0, yaw_acc=0, frame="ENU"):
    #     '''
    #     func->
    #         construct velocity msg
    #     param->
    #         frame: FLU means body-frame forward-left-up
    #                 ENU means world-frame east-north-up
    #     remarks->
    #         uint8 coordinate_frame
    #         uint8 FRAME_LOCAL_NED = 1
    #         uint8 FRAME_LOCAL_OFFSET_NED = 7
    #         uint8 FRAME_BODY_NED = 8
    #         uint8 FRAME_BODY_OFFSET_NED = 9
    #     '''
        
    #     coordinate_frame = 1
    #     if frame == "FLU": 
    #         coordinate_frame = 8
    #     elif frame == "ENU": 
    #         coordinate_frame = 1

    #     target_raw_pose = PositionTarget()
    #     target_raw_pose.header.stamp = rospy.Time.now()

    #     target_raw_pose.coordinate_frame = coordinate_frame 
    #     target_raw_pose.type_mask = (
    #         PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ 
    #         + PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY 
    #         + PositionTarget.FORCE + PositionTarget.IGNORE_YAW + PositionTarget.IGNORE_YAW_RATE
    #     )

    #     target_raw_pose.velocity.x = self.local_velocity.twist.linear.x
    #     target_raw_pose.velocity.y = self.local_velocity.twist.linear.y
    #     target_raw_pose.velocity.z = self.local_velocity.twist.linear.z

    #     target_raw_pose.acceleration_or_force.x = ax
    #     target_raw_pose.acceleration_or_force.z = yaw_acc 

    #     return target_raw_pose

    


    def wait_for_topics(self, timeout):
        """
        wait for simulation to be ready, make sure we're getting topic info
        from all topics by checking dictionary of flag values set in callbacks,
        timeout(int): seconds
        """
        if self.debug:
            rospy.loginfo(self.robot_name + " waiting for subscribed topics to be ready")
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        simulation_ready = False
        for i in xrange(timeout * loop_freq):
            if all(value for value in self.sub_topics_ready.values()):
                simulation_ready = True
                if self.debug:
                    rospy.loginfo(self.robot_name + " simulation topics ready | seconds: {0} of {1}".
                                format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.loginfo(e)

        if not simulation_ready:
            if self.debug:
                rospy.loginfo(self.robot_name + " failed to hear from all subscribed simulation topics | topic ready flags: {0} | timeout(seconds): {1}".
                            format(self.sub_topics_ready, timeout))
    
    def wait_for_landed_state(self, desired_landed_state, timeout, index):
        if self.debug:
            rospy.loginfo(self.robot_name + " waiting for landed state | state: {0}, index: {1}".
                        format(mavutil.mavlink.enums['MAV_LANDED_STATE'][
                            desired_landed_state].name, index))
        loop_freq = 10  # Hz
        rate = rospy.Rate(loop_freq)
        landed_state_confirmed = False
        for i in xrange(timeout * loop_freq):
            if self.extended_state.landed_state == desired_landed_state:
                landed_state_confirmed = True
                if self.debug:
                    rospy.loginfo(self.robot_name + " landed state confirmed | seconds: {0} of {1}".
                                format(i / loop_freq, timeout))
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.loginfo(e)

    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        if self.debug:
            rospy.loginfo(self.robot_name + " setting FCU mode: {0}".format(mode))
        old_mode = self.state.mode
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                if self.debug:
                    rospy.loginfo(self.robot_name + " set mode success | seconds: {0} of {1}".format(
                        i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.loginfo(e)
    
    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        if self.debug:
            rospy.loginfo(self.robot_name + " setting FCU arm: {0}".format(arm))
        old_arm = self.state.armed
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        arm_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.armed == arm:
                arm_set = True
                if self.debug:
                    rospy.loginfo(self.robot_name + " set arm success | seconds: {0} of {1}".format(
                        i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_arming_srv(arm)
                    if not res.success:
                        rospy.logerr(self.robot_name + " failed to send arm command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.loginfo(e)

        if not arm_set:
            if self.debug:
                rospy.loginfo(self.robot_name + " failed to set arm | new arm: {0}, old arm: {1} | timeout(seconds): {2}".
                            format(arm, old_arm, timeout))
        
    
    
    def is_at_position(self, x, y, z, offset):
        """
        func->
            judge if get to the position or not
        params->
            offset(float): meters
        """
        if self.debug:
            rospy.logdebug(
                self.robot_name + " current position | x:{0:.2f}, y:{1:.2f}, z:{2:.2f}".format(
                    self.local_position.pose.position.x, self.local_position.pose.
                    position.y, self.local_position.pose.position.z))

        desired = np.array((x, y, z))
        pos = np.array((self.local_position.pose.position.x,
                        self.local_position.pose.position.y,
                        self.local_position.pose.position.z))
        return np.linalg.norm(desired - pos) < offset


    def get_global_xy_position(self):
        gx = self.local_position.pose.position.x + self.offset[0]
        gy = self.local_position.pose.position.y + self.offset[1]
        return np.array([gx,gy])
    
    def get_local_xYawRateYaw(self):
        vx = self.local_velocity.twist.linear.x
        yaw_rate = self.local_velocity.twist.angular.z
        yaw = self.local_position.pose.orientation.z
        (_, _, yaw) = euler_from_quaternion([   self.local_position.pose.orientation.x,
                                                self.local_position.pose.orientation.y,
                                                self.local_position.pose.orientation.z,
                                                self.local_position.pose.orientation.w])
        return np.array([vx, yaw_rate, yaw])
    #================================== Helper Methods End ==================================


    #================================== PX4 Position Velocity Control Methods Begin ==================================
    def controller_switch(self,ctrl_type="pos"):
        ctrl_dict = PX4_CTRL_DICT[ctrl_type]
        # update the key-value pair to class attribute
        self.__dict__.update(ctrl_dict) 

    def reach_position(self, x, y, z, yaw_degrees, timeout, radius=0.1):
        """
        func->
            reach the position in world frame ENU
        params->
            timeout(int): seconds
            radius(float): meters
        """
        
        # For demo purposes we will lock yaw/heading to north.
        # yaw_degrees = 0  # North
        # yaw = math.radians(yaw_degrees)
        # quaternion = quaternion_from_euler(0, 0, yaw)
        # self.pos.pose.orientation = Quaternion(*quaternion)

        # switch to position control
        self.controller_switch('pos')
        # construct position msg
        yaw = math.radians(yaw_degrees)
        self.command_pos = self.construct_pos_target(x, y, z, yaw, frame="ENU")
        
        if self.debug:
            rospy.loginfo(self.robot_name + " attempting to reach position | x: {0}, y: {1}, z: {2}, yaw: {3} | current position x: {4:.2f}, y: {5:.2f}, z: {6:.2f}".
                        format(x, y, z, yaw_degrees, 
                            self.local_position.pose.position.x,
                            self.local_position.pose.position.y,
                            self.local_position.pose.position.z))


        # does it reach the position in 'timeout' seconds?
        loop_freq = 2  # Hz
        rate = rospy.Rate(loop_freq)
        reached = False
        for i in xrange(timeout * loop_freq):
            if self.is_at_position(self.command_pos.position.x,
                                   self.command_pos.position.y,
                                   self.command_pos.position.z, radius):
                if self.debug:
                    rospy.loginfo(self.robot_name + " position reached | seconds: {0} of {1}".format(
                        i / loop_freq, timeout))
                reached = True
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.loginfo(e)

        if not reached:
            if self.debug:
                rospy.loginfo(self.robot_name + " took too long to get to position | current position x: {0:.2f}, y: {1:.2f}, z: {2:.2f} | timeout(seconds): {3}".
                        format(self.local_position.pose.position.x,
                            self.local_position.pose.position.y,
                            self.local_position.pose.position.z, timeout))


    def moveByVelocityYawrateFLU(self, vx=0, vy=0, vz=0, yaw_rate=0):
        """
        func->
            change the velocity yaw_rate in body frame FLU
        """
        self.controller_switch(ctrl_type="vel")
        self.command_vel = self.construct_vel_target(vx, vy, vz, yaw_rate, frame="FLU")
        if self.debug:
            rospy.loginfo(self.robot_name + " set FLU velocity to vx:{0}, vy:{1}, vz:{2}, yaw_rate:{3}".
                        format(vx, vy, vz, yaw_rate))
        
    def moveByVelocityYawrateENU(self, vx=0, vy=0, vz=0, yaw_rate=0):
        """
        func->
            change the velocity yaw_rate in world frame ENU
        """
        self.controller_switch(ctrl_type="vel")
        self.command_vel = self.construct_vel_target(vx, vy, vz, yaw_rate, frame="ENU")
        if self.debug:
            rospy.loginfo(self.robot_name + " set ENU velocity to vx:{0}, vy:{1}, vz:{2}, yaw_rate:{3}".
                        format(vx, vy, vz, yaw_rate))
    

    # def setAccVelocityXYaw(self, ax=0, yaw_acc=0):
    #     """
    #     func->
    #         change the acceleration in linear x and angular yaw
    #     """
    #     self.controller_switch(ctrl_type="acc")
    #     self.command_acc = self.construct_acc_target(ax, yaw_acc, frame="ENU")
    #     rospy.loginfo("set acceleration to ax:{0}, yaw_acc:{1}".format(ax, yaw_acc))

    def hold_in_place(self):
        """
        func->
            hold in place
        """
        hold_pos = self.local_position
        # switch to position control
        self.controller_switch('pos')

        # maintain position msg
        if hold_pos.pose.position.x > self.local_position.pose.position.x:
            cmd_x = min(hold_pos.pose.position.x, self.local_position.pose.position.x+0.1)
        else:
            cmd_x = max(hold_pos.pose.position.x, self.local_position.pose.position.x-0.1)
            
        if hold_pos.pose.position.y > self.local_position.pose.position.y:
            cmd_y = min(hold_pos.pose.position.y, self.local_position.pose.position.y+0.1)
        else:
            cmd_y = max(hold_pos.pose.position.y, self.local_position.pose.position.y-0.1)

        if hold_pos.pose.position.z > self.local_position.pose.position.z:
            cmd_z = min(hold_pos.pose.position.z, self.local_position.pose.position.z+0.1)
        else:
            cmd_z = max(hold_pos.pose.position.z, self.local_position.pose.position.z-0.1)

        (_, _, cmd_yaw) = euler_from_quaternion([
                                                    hold_pos.pose.orientation.x,
                                                    hold_pos.pose.orientation.y,
                                                    hold_pos.pose.orientation.z,
                                                    hold_pos.pose.orientation.w
                                                    ])
        self.command_pos = self.construct_pos_target(cmd_x, cmd_y, cmd_z, cmd_yaw, frame="ENU")
        if self.debug:
            rospy.loginfo(self.robot_name + " hold in place")

        # hold finish switch to velocity control
        self.controller_switch('vel')



    #================================== PX4 Position Velocity Control Methods End ==================================




    #================================== Debug Functions End ==================================
    def log_topic_vars(self):
        """log the state of topic variables"""
        rospy.loginfo("========================")
        rospy.loginfo("===== " + self.robot_name + " topic values =====")
        rospy.loginfo("========================")
        rospy.loginfo("extended_state:\n{}".format(self.extended_state))
        rospy.loginfo("========================")
        rospy.loginfo("local_position:\n{}".format(self.local_position))
        rospy.loginfo("========================")
        rospy.loginfo("local_velocity:\n{}".format(self.local_velocity))
        rospy.loginfo("========================")
        rospy.loginfo("state:\n{}".format(self.state))
        rospy.loginfo("========================")

    #================================== Debug Functions End ==================================



    #================================== UAV Command Functions Begin ==================================
    def takeoff(self, height):

        # make sure the simulation is ready
        self.wait_for_topics(60)
        # log topics for debug
        if self.debug:
            self.log_topic_vars()

        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   10, -1)
        
        self.set_mode("OFFBOARD", 5)
        self.set_arm(True, 10)
        if self.debug:
            rospy.loginfo(self.robot_name + ' Ready to take off as height {0}'.format(height))
        self.reach_position(0, 0, height, 0, 15)


    def land(self):

        self.set_mode("AUTO.LAND", 5)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   45, 0)
        self.set_arm(False, 10)
        # log topics for debug
        if self.debug:
            self.log_topic_vars()
    
    def home(self, maintain_height, mid_height=10):

        self.reach_position(self.local_position.pose.position.x, 
                            self.local_position.pose.position.y, 
                            mid_height, 0, 15, radius=0.1)
        self.reach_position(0, 0, mid_height, 0, 15, radius=0.1)
        self.reach_position(0, 0, maintain_height, 0, 15, radius=0.1)
        

    #================================== UAV Command Functions End ==================================




if __name__ == "__main__":
    # test
    rospy.init_node('test_px4_controller')
    traject_iris = None
    traject_UST10LX = None

    def control_iris():
        global traject_iris
        px4ctl_iris = PX4OffboardController(robot_name='iris', offset=[7,0], debug=True)  
        px4ctl_iris.takeoff(4.0)
        rospy.sleep(rospy.Duration(1.0))
        px4ctl_iris.set_record(True)
        px4ctl_iris.moveByVelocityYawrateFLU(1,0,0,0.2)
        rospy.sleep(rospy.Duration(10.0))
        px4ctl_iris.set_record(False)
        px4ctl_iris.home(maintain_height=0.5)
        rospy.sleep(rospy.Duration(1.0))
        px4ctl_iris.land()
        traject_iris = px4ctl_iris.get_trajectory()


    def control_UST10LX():  
        global traject_UST10LX
        px4ctl_UST10LX = PX4OffboardController(robot_name='iris_UST10LX', offset=[0,0], debug=True)
        px4ctl_UST10LX.takeoff(5.0)
        print(px4ctl_UST10LX.get_local_xyaw_velocity())
        rospy.sleep(rospy.Duration(1.0))
        px4ctl_UST10LX.set_record(True)
        px4ctl_UST10LX.moveByVelocityYawrateFLU(1,0,0,-0.2)
        rospy.sleep(rospy.Duration(10.0))
        print(px4ctl_UST10LX.get_local_xyaw_velocity())
        px4ctl_UST10LX.hold_in_place()
        rospy.sleep(rospy.Duration(10.0))
        px4ctl_UST10LX.set_record(False)
        px4ctl_UST10LX.home(maintain_height=0.5)
        rospy.sleep(rospy.Duration(1.0))
        px4ctl_UST10LX.land()
        traject_UST10LX = px4ctl_UST10LX.get_trajectory()


    thread1 = Thread(target=control_iris, args=())
    thread2 = Thread(target=control_UST10LX, args=())
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(traject_iris[:,0], traject_iris[:,1], label='iris trajectory', color='red', linestyle='-')
    ax.plot(traject_UST10LX[:,0], traject_UST10LX[:,1], label='UST10LX trajectory', color='blue', linestyle='--')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectory in 2D')
    ax.legend()
    plt.show()

        
