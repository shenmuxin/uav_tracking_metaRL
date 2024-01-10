#! /usr/bin/env python
# coding :utf-8

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# check import path
# for path in sys.path:
#     print(path)

from utils.offb_fly import PX4OffboardController
from envs.my_world import World

import rospy
from sensor_msgs.msg import LaserScan
from mavros_msgs.msg import PositionTarget
from sitl_study.msg import Reward
from sitl_study.msg import State

import numpy as np
import math
import angles

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


class CommandFilter:
    """Command Filter is used to smooth the command"""
    def __init__(self, window_size, input_dim=2):
        self.window_size = window_size
        self.input_dim = input_dim
        self.command_window = [[] for _ in range(input_dim)]

    def filter_command(self, command):
        for i in range(self.input_dim):
            self.command_window[i].append(command[i])

            # get rigid of old cmd
            if len(self.command_window[i]) > self.window_size:
                self.command_window[i].pop(0)

            # get out of max and min cmd
            if len(self.command_window[i]) == self.window_size:
                self.command_window[i].remove(max(self.command_window[i]))
                self.command_window[i].remove(min(self.command_window[i]))

        # average cmd
        filtered_command = [np.mean(self.command_window[i]) for i in range(self.input_dim)]

        return filtered_command

class GameAvoid:
    def __init__(self, safe_points = [[0,0], [7,0]]):
        """
            safe_points: [[0,0], [7,0]] for default
        """
        self.cmdfilter = CommandFilter(window_size=5)

        # specify the point of pursuer and avoider
        self.safe_points = safe_points
        self.pursuer_ctl = PX4OffboardController(robot_name='iris_UST10LX', offset=safe_points[0], debug=False)
        self.avoider_ctl = PX4OffboardController(robot_name='iris', offset=safe_points[1], debug=False)

        # whether is normalized
        self.is_normalized = False

        # specify maximum action
        self.action_bound = [2.0, 1.0]

        # specify fly height
        self.avoider_height = 3
        self.pursuer_height = 5

        # specify the catch limit
        self.catch_limit = 0.5

        # specify out limit
        self.out_limit = 20 


        # specify crash limit
        self.crash_limit = 0.3

        # instance world
        self.world = World(safe_points, [2, 2], [8,10])


        # initialize position
        self.pursuer_pos = self.pursuer_ctl.get_global_xy_position()
        self.avoider_pos = self.avoider_ctl.get_global_xy_position()

        # whether begin flight control
        self.flight_flag = False

        # initialize step count
        self.step_count = 0

        # initialize LaserScan
        self.scan = LaserScan()

        # self.hold_flag = False
        self.hold_able = False
        # record pursuer's last cmd time
        self.last_cmd_time = rospy.Time.now()


        # scan subscriber
        self.scanSub = rospy.Subscriber(self.pursuer_ctl.robot_name + "_0/scan_filtered", LaserScan, self._scanCB)

        # reward publisher
        self.rewardPub = rospy.Publisher("reward", Reward, queue_size=1)
        # state publisher
        self.statePub = rospy.Publisher("rl_state", State, queue_size=1)


        # pursuer hold timer
        self._pursuerHoldTimer = rospy.Timer(rospy.Duration(0.05), self._hold)
        # avoider ctrl timer
        self._avoiderCtrlTimer = rospy.Timer(rospy.Duration(0.05), self._avoiderCtrl)




    #========================================= CallBack Functions Begin =========================================
    def _scanCB(self, msg):
        self.scan = msg
    
    #========================================= CallBack Functions End =========================================
    
    #========================================= Timer Functions Begin =========================================
    def _hold(self, event):
        """
        func->
            if no cmd is sent, pursuer hold in place
        """

        # make sure during flight ctrl
        if self.flight_flag == True:
            if self.hold_able == False:
                return
            if self.hold_able == True:
                if (rospy.Time.now() - self.last_cmd_time >= rospy.Duration(0.1)):
                    self.pursuer_ctl.hold_in_place()

    def _avoiderCtrl(self, event):
        """
        func->
            control avoider flight pattern
        """
        if self.flight_flag == True:
            # 假定avoider直线飞行
            self.avoider_ctl.moveByVelocityYawrateFLU(vx=0.5, vy=0, vz=0, yaw_rate=0)

            

    #========================================= Timer Functions End =========================================


    #========================================= Tool Functions Begin =========================================
    def start(self):
        """
        func->
            take off the uav and reset the environment
        """
        # generate world
        self.create_world()

        # takeoff
        self.avoider_ctl.takeoff(self.avoider_height)
        self.pursuer_ctl.takeoff(self.pursuer_height)
        rospy.sleep(rospy.Duration(1.0))
        rospy.loginfo("Begin Game")
        # begin flight control
        self.flight_flag = True
        # clear step count
        self.step_count = 0
    



        if self.is_normalized:
            initial_state = self.curr_state_normalized()
        else:
            initial_state = self.curr_state()
        
        return initial_state 


    def reset(self):
        """
        func->
            reset the uav and reset the environment
        """
        # stop flight control
        self.flight_flag = False

        # clear velocity cmd
        self.avoider_ctl.moveByVelocityYawrateFLU()
        self.pursuer_ctl.moveByVelocityYawrateFLU()
        # fly home
        self.pursuer_ctl.home(maintain_height=self.pursuer_height)
        self.avoider_ctl.home(maintain_height=self.avoider_height)

        # TODO: add obstacle reset

        if self.is_normalized:
            initial_state = self.curr_state_normalized()
        else:
            initial_state = self.curr_state()

        # start flight control
        self.flight_flag = True
        # clear step count
        self.step_count = 0

        return initial_state 


    def step(self, vx=0, yaw_rate=0, time_step=0.1):
        """
        return->
            next state
            reward
            terminated
            truncated
        """
        # initial
        terminated = False
        truncated = False
        one_step_reward = 0

        # update some parameters
        self.step_count += 1
        self.hold_able = False

        # old distance
        last_distance = self.calculate_distance()
        # for debug
        # print("Distance is %.2f"%last_distance)

        # send control command for time_step period
        time = rospy.Time.now()
        self.pursuer_ctl.moveByVelocityYawrateFLU(vx=vx, yaw_rate=yaw_rate)
        while (rospy.Time.now() - time < rospy.Duration(time_step)):
            out_indicator = self.is_out()
            crash_indicator = self.is_crashed()
            if (out_indicator == True) or (crash_indicator == True):
                break

        # clear velocity control
        self.pursuer_ctl.moveByVelocityYawrateFLU()
        # new distance
        curr_distance = self.calculate_distance()
        # update hold action
        self.hold_able = True
        


        #>>>>>>>>>>>>>>>>>>> calculate reward <<<<<<<<<<<<<<<<<<<<
        # distance reward
        distance_reward = (last_distance - curr_distance)*(5/time_step)

        # step punish reward
        step_punish_reward = -self.step_count * 0.01

        # angular reward
        alpha = self.calculate_angular()
        if (alpha > math.pi / 8):
            angular_reward = (math.pi / 8 - alpha) * 5 / math.pi
        else:
            angular_reward = (math.pi / 8 - alpha) * 32 / math.pi

        # reward initialize
        catch_reward = 0
        crash_reward = 0
        out_punish_reward = 0

        # if successfully catch
        if curr_distance < self.catch_limit:  
            catch_reward = 200
            terminated = True
            print("====== Successfully catch ======")


        # if crashed
        if crash_indicator == True:     # if pursuer is crashed
            truncated = True
            crash_reward = self.laser_crashed_reward

        # if out of limit
        if out_indicator == True:
            truncated = True
            out_punish_reward = -100


        one_step_reward = distance_reward + \
                       step_punish_reward + \
                       angular_reward + \
                       catch_reward + \
                       crash_reward + \
                       out_punish_reward
        #>>>>>>>>>>>>>>>>>>> calculate reward <<<<<<<<<<<<<<<<<<<<
        
        # publish reward
        reward_msg = Reward()
        reward_msg.header.stamp = rospy.Time.now()
        reward_msg.distance_reward = distance_reward
        reward_msg.step_punish_reward = step_punish_reward
        reward_msg.angular_reward = angular_reward
        reward_msg.catch_reward = catch_reward
        reward_msg.crash_reward = crash_reward
        reward_msg.out_punish_reward = out_punish_reward

        self.rewardPub.publish(reward_msg)

        if self.is_normalized:
            next_state = self.curr_state_normalized()
        else:
            next_state = self.curr_state()
        return next_state, np.array([one_step_reward]), terminated, truncated

    def step_with_cmd_filter(self, vx=0, yaw_rate=0, time_step=0.1):
        """
        return->
            next state
            reward
            terminated
            truncated
        """
        # initial
        terminated = False
        truncated = False
        one_step_reward = 0

        # update some parameters
        self.step_count += 1
        self.hold_able = False

        # old distance
        last_distance = self.calculate_distance()

        # send control command for time_step period
        time = rospy.Time.now()

        cmd = self.cmdfilter.filter_command([vx, yaw_rate])
        self.pursuer_ctl.moveByVelocityYawrateFLU(vx=cmd[0], yaw_rate=cmd[1])
        while (rospy.Time.now() - time < rospy.Duration(time_step)):
            out_indicator = self.is_out()
            crash_indicator = self.is_crashed()
            if (out_indicator == True) or (crash_indicator == True):
                break

        # clear velocity control
        self.pursuer_ctl.moveByVelocityYawrateFLU()
        # new distance
        curr_distance = self.calculate_distance()
        # update hold action
        self.hold_able = True
        


        #>>>>>>>>>>>>>>>>>>> calculate reward <<<<<<<<<<<<<<<<<<<<
        # distance reward
        distance_reward = (last_distance - curr_distance)*(5/time_step)

        # step punish reward
        step_punish_reward = -self.step_count * 0.01

        # angular reward
        alpha = self.calculate_angular()
        if (alpha > math.pi / 8):
            angular_reward = (math.pi / 8 - alpha) * 5 / math.pi
        else:
            angular_reward = (math.pi / 8 - alpha) * 32 / math.pi

        # reward initialize
        catch_reward = 0
        crash_reward = 0
        out_punish_reward = 0

        # if successfully catch
        if curr_distance < self.catch_limit:  
            catch_reward = 200
            terminated = True
            print("====== Successfully catch ======")


        # if crashed
        if crash_indicator == True:     # if pursuer is crashed
            truncated = True
            crash_reward = self.laser_crashed_reward

        # if out of limit
        if out_indicator == True:
            truncated = True
            out_punish_reward = -100


        one_step_reward = distance_reward + \
                       step_punish_reward + \
                       angular_reward + \
                       catch_reward + \
                       crash_reward + \
                       out_punish_reward
        #>>>>>>>>>>>>>>>>>>> calculate reward <<<<<<<<<<<<<<<<<<<<
        
        # publish reward
        reward_msg = Reward()
        reward_msg.header.stamp = rospy.Time.now()
        reward_msg.distance_reward = distance_reward
        reward_msg.step_punish_reward = step_punish_reward
        reward_msg.angular_reward = angular_reward
        reward_msg.catch_reward = catch_reward
        reward_msg.crash_reward = crash_reward
        reward_msg.out_punish_reward = out_punish_reward

        self.rewardPub.publish(reward_msg)

        if self.is_normalized:
            next_state = self.curr_state_normalized()
        else:
            next_state = self.curr_state()
        return next_state, np.array([one_step_reward]), terminated, truncated



    #========================================= Tool Functions End =========================================








    #========================================= Helper Method Begin =========================================
    def is_crashed(self):
        """
        func->
            judge whether pursuer is crashed
        """

        truncated = False
        self.laser_crashed_reward = 0
        self.crash_index = -1

        for i in range(len(self.scan.ranges)):
            if self.scan.ranges[i] < 3*self.crash_limit:
                self.laser_crashed_reward = min(-10, self.laser_crashed_reward)
            if self.scan.ranges[i] < 2*self.crash_limit:
                self.laser_crashed_reward = min(-25.0, self.laser_crashed_reward)
            if self.scan.ranges[i] < self.crash_limit:
                self.laser_crashed_reward = -1000.0
                truncated = True
                self.crash_index = i
                break
        # for debug
        if truncated:
            print("====== Crashed ======")
        return truncated

    def is_out(self):
        """
        func->
            judge whether avoider or pursuer is out of range limit
        """
    

        truncated = False
        self.avoider_pos = self.avoider_ctl.get_global_xy_position()
        self.pursuer_pos = self.pursuer_ctl.get_global_xy_position()
        # if avoider successfully escape
        if (self.avoider_pos[0] < - self.out_limit or self.avoider_pos[0] > self.out_limit or self.avoider_pos[1] < -self.out_limit or self.avoider_pos[1] > self.out_limit):
            truncated = True
        # if pursuer out of safe space
        elif (self.pursuer_pos[0] < -self.out_limit or self.pursuer_pos[0] > self.out_limit or self.pursuer_pos[1] < -self.out_limit or self.pursuer_pos[1] > self.out_limit):
            truncated = True
        # for debug
        if truncated:
            print("====== Avoider Out ======")
        return truncated
    
    def calculate_distance(self):
        """
        func->
            calculate relative distance
        """
        self.pursuer_pos = self.pursuer_ctl.get_global_xy_position()
        self.avoider_pos = self.avoider_ctl.get_global_xy_position()

        # calculate Euclidean distance
        dis = np.linalg.norm(self.pursuer_pos - self.avoider_pos)
        return dis
    
    def calculate_angular(self):
        """
        func->
            calculate connection line motion angular diff
        """
        _, _, p_yaw = self.pursuer_ctl.get_local_xYawRateYaw()
        p_x, p_y = self.pursuer_ctl.get_global_xy_position()
        a_x, a_y = self.avoider_ctl.get_global_xy_position()
        dis_angle = math.atan2(a_y - p_y, a_x - p_x)
        connection_line_motion_diff = math.fabs(angles.shortest_angular_distance(p_yaw, dis_angle))
        return connection_line_motion_diff
    
    def curr_state(self):
        """
            35 laser ranges
            1  p_x
            1  p_y
            1  a_x
            1  a_y
            1  p_vx
            1  p_yaw_rate
            1  p_yaw
            1  a_vx
            1  a_yaw_rate
            1  a_yaw
            1  relative_distance
            1  connection_line_motion_diff
        """
        # ranges msg
        state = list(self.scan.ranges)

        p_x, p_y = self.pursuer_ctl.get_global_xy_position()
        a_x, a_y = self.avoider_ctl.get_global_xy_position()
        # position msg
        state.append(p_x)
        state.append(p_y)
        state.append(a_x)
        state.append(a_y)
        # pose msg
        p_vx, p_yaw_rate, p_yaw = self.pursuer_ctl.get_local_xYawRateYaw()
        a_vx, a_yaw_rate, a_yaw = self.avoider_ctl.get_local_xYawRateYaw()
        
        state.append(p_vx)
        state.append(p_yaw_rate)
        state.append(p_yaw)
        state.append(a_vx)
        state.append(a_yaw_rate)
        state.append(a_yaw)

        # relative distance msg
        relative_distance = self.calculate_distance()
        state.append(relative_distance)

        # relative distance angular diff
        connection_line_motion_diff = self.calculate_angular()
        state.append(connection_line_motion_diff)

        # publish state
        state_msg = State()
        state_msg.header.stamp = rospy.Time.now()
        state_msg.curr_state = state
        self.statePub.publish(state_msg)
        
        return np.array(state)
    

    def curr_state_normalized(self):
        """
            35 laser ranges
            1  p_x
            1  p_y
            1  a_x
            1  a_y
            1  p_vx
            1  p_yaw_rate
            1  p_yaw [-pi, pi]
            1  a_vx
            1  a_yaw_rate
            1  a_yaw [-pi, pi]
            1  relative_distance
            1  connection_line_motion_diff
        """
        # TODO:add obstacles
        # ranges msg
        # state = list(self.scan.ranges)

        state = []

        # position msg
        p_x, p_y = self.pursuer_ctl.get_global_xy_position()
        a_x, a_y = self.avoider_ctl.get_global_xy_position()
        # normalize position
        p_xn = p_x / self.out_limit
        p_yn = p_y / self.out_limit
        a_xn = a_x / self.out_limit
        a_yn = a_y / self.out_limit
        state.append(p_xn)
        state.append(p_yn)
        state.append(a_xn)
        state.append(a_yn)

        # pose msg
        p_vx, p_yaw_rate, p_yaw = self.pursuer_ctl.get_local_xYawRateYaw()
        a_vx, a_yaw_rate, a_yaw = self.avoider_ctl.get_local_xYawRateYaw()
        # normalize pose
        p_vxn = p_vx /self.action_bound[0]
        p_yaw_raten = p_yaw_rate / self.action_bound[1]
        p_yawn = p_yaw / math.pi
        a_vxn = a_vx / self.action_bound[0]
        a_yaw_raten = a_yaw_rate / self.action_bound[1]
        a_yawn = a_yaw / math.pi
        
        state.append(p_vxn)
        state.append(p_yaw_raten)
        state.append(p_yawn)
        state.append(a_vxn)
        state.append(a_yaw_raten)
        state.append(a_yawn)

        # relative distance msg
        relative_distance = self.calculate_distance() / (math.sqrt(2) * self.out_limit)
        state.append(relative_distance)

        # relative distance angular diff
        connection_line_motion_diff = self.calculate_angular() / math.pi
        state.append(connection_line_motion_diff)

        # publish state
        state_msg = State()
        state_msg.header.stamp = rospy.Time.now()
        state_msg.curr_state = state
        self.statePub.publish(state_msg)
        
        return state
    
    def set_trajectory_record(self, flag):
        if flag == True:
            # clear trajectories
            self.avoider_ctl.clear_traject()
            self.pursuer_ctl.clear_traject()
        # begin to record
        self.avoider_ctl.set_record(flag)
        self.pursuer_ctl.set_record(flag)
    
    def get_trajectory(self):
        """get trajectory data, store as a tuple, each element is an numpy array"""

        return (self.avoider_ctl.get_trajectory(), self.pursuer_ctl.get_trajectory())


    def plot_fly_trajectory2d(self):
        avoider_trj = self.avoider_ctl.get_trajectory()
        pursuer_trj = self.pursuer_ctl.get_trajectory()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(avoider_trj[:,0], avoider_trj[:,1], label='avoider trajectory', color='red', linestyle='-')
        ax.plot(pursuer_trj[:,0], pursuer_trj[:,1], label='pursuer trajectory', color='blue', linestyle='--')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Trajectory in 2D')
        ax.set_ylim(-3,3)
        ax.legend()
        plt.show()
    
    def plot_fly_trajectory3d(self):
        avoider_trj = self.avoider_ctl.get_trajectory()
        pursuer_trj = self.pursuer_ctl.get_trajectory()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(avoider_trj[:,0], avoider_trj[:,1], avoider_trj[:,2], label='avoider trajectory', color='red', linestyle='-')
        ax.plot(pursuer_trj[:,0], pursuer_trj[:,1], pursuer_trj[:,2], label='pursuer trajectory', color='blue', linestyle='--')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory in 3D')
        ax.legend()
        plt.show()


    def create_world(self):
        self.world.generate_cylinders()
    
    def clear_world(self):
        self.world.clear()
    
    def get_map(self):
        return self.world.get_tree_map()








    #========================================= Helper Method End =========================================


    #========================================= Debug Function Begin =========================================
    def log_topic_vars(self):
        """log the state of topic variables"""
        rospy.loginfo("========================")
        rospy.loginfo("===== " + self.pursuer_ctl.robot_name + " topic values =====")
        rospy.loginfo("========================")
        rospy.loginfo("laser_scan:\n{}".format(self.scan))
        if self.is_normalized:
            rospy.loginfo("current_state:\n{}".format(self.curr_state_normalized()))
        else:
            rospy.loginfo("current_state:\n{}".format(self.curr_state()))


    #========================================= Debug Function End =========================================



if __name__ == "__main__":
    
    pass


