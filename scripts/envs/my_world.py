#! /usr/bin/env python
#-*- coding: UTF-8 -*- 

import rospy
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, DeleteModel,DeleteModelRequest
from tf.transformations import quaternion_from_euler


import xml.etree.ElementTree as ET
import numpy as np
import math
import os

import matplotlib.pyplot as plt

class World:

    def __init__(self, safe_points, safe_radius, cylinder_num):
        """
        func->
            The world class used to generate models, such as ball and cylinders
        params->
            safe_points: specify the safe point respectively, such as [[0,0], [7,0]]
            safe_radius: specify the safe radius respectively, such as [1, 1]
            cylinder_num: specify the range of cylinder num, such as [120, 150]
        """

        # constant
        self.safe_points = safe_points
        self.safe_radius = safe_radius
        self.cylinder_num = cylinder_num
        # length of the square space
        self.xlimit = [-6, 6]
        self.ylimit = [-6, 6]
        # minimum distance between cylinders
        self.min_distance = 1.5

        # initialize cylinder list
        self.cylinder_name_list = []
        self.cylinder_position_list = []

        # service client
        self.spawnModelSdfClient = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.deleteModelClient = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        # wait for service
        self.spawnModelSdfClient.wait_for_service()
        self.deleteModelClient.wait_for_service()

        # read XML file
        self.url = os.path.dirname(os.path.realpath(__file__))

        self.xmltree = ET.parse(self.url + '/cylinder/model.sdf')



    def generate_cylinders(self):
        """
        func->
            generate cylinders in gazebo
        """

        # randomly change the radius of cylinders
        root = self.xmltree.getroot()
        

        
    
        # reset cylinder
        msg = SpawnModelRequest()
        
        msg.initial_pose.position.z = 4
        msg.initial_pose.orientation.x = 0
        msg.initial_pose.orientation.y = 0
        msg.initial_pose.orientation.z = 0
        msg.initial_pose.orientation.w = 1

        # generate cylinder num randomly
        cylinder_nums = np.random.randint(self.cylinder_num[0], self.cylinder_num[1])

        for i in range(0, cylinder_nums):

            # set radius
            radius = round(np.random.uniform(0.3, 0.6), 2)
            for cylinder in root.iter('cylinder'):
                radius_element = cylinder.find('radius')
                if radius_element is not None:
                    radius_element.text = str(radius)

            xml_content = ET.tostring(root, encoding='unicode')

            # set px py
            msg.model_name = "cylinder_" + str(i)
            self.cylinder_name_list.append(msg.model_name)
            px = np.random.uniform(self.xlimit[0], self.xlimit[1])
            py = np.random.uniform(self.ylimit[0], self.ylimit[1])
            
            while not self.check_safe(px, py, radius):
                px = np.random.uniform(self.xlimit[0], self.xlimit[1])
                py = np.random.uniform(self.ylimit[0], self.ylimit[1])
            self.cylinder_position_list.append((px, py, radius))

            # specify the msg
            msg.model_xml = xml_content
            msg.initial_pose.position.x = px
            msg.initial_pose.position.y = py

            try:
                if(self.spawnModelSdfClient.call(msg).success ==True):
                    pass
            except:
                pass
        rospy.loginfo("==================================")
        rospy.loginfo(">>>>>> Generate Cylinders >>>>>>")
    

    def clear(self):
        """
        func->
            clear world
        """
        msg = DeleteModelRequest()
        # call service

        try:
            # delete cylinder
            for item in self.cylinder_name_list:
                msg.model_name = item
                self.deleteModelClient.call(msg)
            self.cylinder_name_list = []
            self.cylinder_position_list = []
            # delete ball
            # msg.model_name = self.target_model_name
            # self.deleteModelClient.call(msg)
        except:
            pass
        rospy.loginfo("==================================")
        rospy.loginfo(">>>>>> Clear Everything >>>>>>")


    # tool functions
    def set_cylinder_rate(self, cylinder_rate):
        self.cylinder_rate = cylinder_rate

    def check_safe(self, x, y, r):
        """
        func->
            check whether (x, y) falls into safe space, whether satisfy the minimum distance constraint
        return->
            True - if not fall into
            False - if fall into
        """
        # prevent cylinder in safe point
        for i in range(0, len(self.safe_points)):
            distance = math.sqrt((self.safe_points[i][0] - x)**2 + (self.safe_points[i][1] - y)**2) - r
            if distance < self.safe_radius[i]:
                return False
        # maintain minimum dis in cylinders
        for (px, py, pr) in self.cylinder_position_list:
            distance = math.sqrt((x-px)**2 + (y-py)**2) - r - pr
            if distance < self.min_distance:
                return False
        # # prevent cylinder in avoider flight way
        # if y >= 0:
        #     if (x > self.safe_points[1][0]) and (y-r-1<0):
        #         return False
        # elif y < 0:
        #     if (x > self.safe_points[1][0]) and (y+r+1>0):
        #         return False
        return True
 
    def get_tree_map(self):
        return np.array(self.cylinder_position_list)
    
    def plot_tree_map(self):
        tree_map = self.get_tree_map()
        # print(tree_map)
        x = tree_map[:,0]
        y = tree_map[:,1]
        distances_in_meters = tree_map[:,2]

        dpi = 100  # 每英寸点数
        inches_per_meter = 0.0254  # 1米 = 0.0254英寸
        radii_in_pixels = [distance * dpi  for distance in distances_in_meters]

        plt.scatter(x, y, s=radii_in_pixels, c='black', marker='o')

        plt.title('Scatter Plot with Custom Radii')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        plt.legend()

        plt.show()


if __name__ == '__main__':

    rospy.init_node("test_node")
    
    world = World([[0, 0], [7, 0]], [2, 2], [10, 12])
    
    world.generate_cylinders()
    world.plot_tree_map()
    world.clear()
    # rospy.sleep(rospy.Duration(10))
