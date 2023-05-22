#!/usr/bin/env python3

import rospy
from datetime import datetime
import rospkg
from geometry_msgs.msg import PoseStamped
from copy import deepcopy
import gym
import numpy as np
import math
from gym import spaces
import bimanual_handover.syn_grasp_gen as sgg
import sensor_msgs.point_cloud2 as pc2
from moveit_msgs.msg import MoveItErrorCodes
from moveit_commander import MoveItCommanderException

class SimpleEnv(gym.Env):

    def __init__(self, fingers, pc, ps = None):
        super().__init__()
        self.action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype = np.float64) # + decision if finished, limits found through manual testing
        self.observation_space = spaces.Dict({"finger_contacts": spaces.MultiBinary(5)})#"finger_joints": spaces.Box(low = -1, high = 1, shape = (22,), dtype = np.float64), spaces.Box(-1.134, 1.658, shape = (22,), dtype = np.float64), "finger_contacts": spaces.MultiBinary(5)})# (last_action, finger_joints, finger_contacts) {"last_action": spaces.Box(low = np.array([-1.5, -0.65, -1.25]), high = np.array([1.5, 0.65, 1.25]), dtype = np.float64),
        self.fingers = fingers
        self.pc = pc
        self.ps = ps
        self.pca_con = sgg.SynGraspGen()
        self.last_joints = None
        self.debug_pub = rospy.Publisher('env_debug', PoseStamped, queue_size = 1)
        self.time = datetime.now().strftime("%d_%m_%Y_%H_%M")
        self.log_file = open(rospkg.RosPack().get_path('bimanual_handover') + "/logs/log" + self.time + ".txt", 'w')
        self.joint_order = self.fingers.get_active_joints()
        self.reset_timer = 0

    def step(self, action):
        reward = 0
        terminated = False
        scaled_action = np.array([action[0] * 1.5, action[1] * 0.65, action[2] * 1.25])
        try:
            result = self.pca_con.move_joint_config(scaled_action)
        except MoveItCommanderException as e:
            print(e)
            reward = -0.2
        if not result is MoveItErrorCodes.SUCCESS:
            reward = -0.2
        contact_points = self.__contact_points()
        current_joints = np.array(self.fingers.get_current_joint_values())
        normalized_joints = self.__normalize_joints(current_joints)
        observation = {}
#        observation["last_action"] = action
        #observation["finger_joints"] = normalized_joints
        observation["finger_contacts"] = contact_points
        ff = ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1']
        mf = ['rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1']
        rf = ['rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1']
        lf = ['rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1']
        th = ['rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1']
        finger_list = [ff, mf, rf, lf, th]
        if reward >= 0:
            joint_diff = 0
            for c in range(len(contact_points)):
                if contact_points[c] == 0:
                    finger = finger_list[c]
                    new_joints_finger = [normalized_joints[self.joint_order.index(joint)] for joint in finger]
                    last_joints_finger = [self.last_joints[self.joint_order.index(joint)] for joint in finger]
                    for i in range(len(last_joints_finger)):
                        if new_joints_finger[i] >= last_joints_finger[i]:
                            joint_diff += math.dist([new_joints_finger[i]], [last_joints_finger[i]])
                        else:
                            joint_diff += - math.dist([new_joints_finger[i]], [last_joints_finger[i]])
            '''
            for i in range(len(self.last_joints)):
                if new_joints[i] >= self.last_joints[i]:
                    joint_diff += math.dist([new_joints[i]], [self.last_joints[i]])
                else:
                    joint_diff += - math.dist([new_joints[i]], [self.last_joints[i]])
            '''
            reward = 0.03 * joint_diff
            reward += np.sum(contact_points) * 0.2
            if np.sum(contact_points) == 5:
                #reward = 1
                reward += 0.2
#                terminated = True
        '''
        if np.sum(contact_points) > 0 and reward != -1:
            reward = np.sum(contact_points, dtype = np.float64)
            if np.sum(contact_points) == 5:
                reward += 1
                terminated = True
        elif reward != -1:
            joint_diff = 0
            for i in range(len(self.last_joints)):
                if new_joints[i] >= self.last_joints[i]:
                    joint_diff += math.dist([new_joints[i]], [self.last_joints[i]])
                else:
                    joint_diff += - math.dist([new_joints[i]], [self.last_joints[i]])
            reward = 0.03 * joint_diff
            # 0.2 * np.sum(contact_points)# reward: 0.03 * distance between new and old joint values during session, 1 or 0 if enough contact points
        '''
        self.last_joints = normalized_joints 
        truncated = False
        info = {}
        print(scaled_action)
        print(reward)
        self.log_file.write("{}, {} \n".format(action, reward))
        self.reset_timer += 1
        if self.reset_timer == 100:
            terminated = True
        return observation, reward, terminated, info

    def reset(self):
        self.log_file.close()
        self.log_file = open(rospkg.RosPack().get_path('bimanual_handover') + "/logs/log"+ self.time + ".txt", 'a')
        self.reset_timer = 0
        self.fingers.set_named_target('open')
        self.fingers.go()
        self.pca_con.set_initial_hand_joints()
        contact_points = self.__contact_points()
        observation = {}
#        observation["last_action"] = np.array([0, 0, 0], dtype = np.float64)
        current_joints = np.array(self.fingers.get_current_joint_values())
        normalized_joints = self.__normalize_joints(current_joints)
        #observation["finger_joints"] = normalized_joints
        observation["finger_contacts"] = contact_points
        self.last_joints = normalized_joints
        info = {}
        return observation#, info

    def render(self):
        return

    def close(self):
        self.log_file.close()
        return

    def __normalize_joints(self, joints):
        '''
        J2: 0-pi/2
        J1: 0-pi/2
        J3: -15*pi/180 - pi/2
        J4: -20/180*pi - 20/180*pi
        LFJ5: 0 - 45/180*pi
        THJ1: -15/180*pi - pi/2
        THJ2: -40/180*pi - 40/180*pi
        THJ3: -12/180*pi - 12/180 * pi
        THJ4: 0 - 70/180*pi
        THJ5: -60/180pi - 60/180*pi
        '''
        ranges = {'rh_FFJ1': [0, math.pi/2], 'rh_FFJ2': [0, math.pi/2], 'rh_FFJ3': [-15/180*math.pi, math.pi/2], 'rh_FFJ4': [-20/180 * math.pi, 20/180 * math.pi], 'rh_MFJ1': [0, math.pi/2], 'rh_MFJ2': [0, math.pi/2], 'rh_MFJ3': [-15/180*math.pi, math.pi/2], 'rh_MFJ4': [-20/180 * math.pi, 20/180 * math.pi], 'rh_RFJ1': [0, math.pi/2], 'rh_RFJ2': [0, math.pi/2], 'rh_RFJ3': [-15/180*math.pi, math.pi/2], 'rh_RFJ4': [-20/180 * math.pi, 20/180 * math.pi], 'rh_LFJ1': [0, math.pi/2], 'rh_LFJ2': [0, math.pi/2], 'rh_LFJ3': [-15/180*math.pi, math.pi/2], 'rh_LFJ4': [-20/180 * math.pi, 20/180 * math.pi], 'rh_LFJ5': [0, 45/180 * math.pi], 'rh_THJ1': [-15/180*math.pi, math.pi/2], 'rh_THJ2': [-40/180*math.pi, 40/180*math.pi], 'rh_THJ3': [-12/180*math.pi, 12/180*math.pi], 'rh_THJ4': [0, 80/180*math.pi], 'rh_THJ5': [-60/180*math.pi, 60/180*math.pi]} 
        normalized_joints = []
        for i in range(len(self.joint_order)):
            limits = ranges[self.joint_order[i]]
            shift = (limits[1] + limits[0])/2
            factor = limits[1] - shift
            normalized_joints.append((joints[i]-shift)/factor)
        normalized_joints = np.array(normalized_joints)
        return normalized_joints

    def __contact_points(self):
        cylinder = self.ps.get_objects(['can'])['can']
        h = cylinder.primitives[0].dimensions[0]
        r = cylinder.primitives[0].dimensions[1]
        pose = cylinder.pose
        p1 = deepcopy(pose.position)
        p1.z += h/2
        p1 = np.array([p1.x, p1.y, p1.z])
        p2 = deepcopy(pose.position)
        p2.z += -h/2
        p2 = np.array([p2.x, p2.y, p2.z])

        debug_pose = PoseStamped()
        debug_pose.header.frame_id = 'base_footprint'
        debug_pose.pose.position.x = p2[0]
        debug_pose.pose.position.y = p2[1]
        debug_pose.pose.position.z = p2[2]
        self.debug_pub.publish(debug_pose)

        tip_links = ['rh_ff_biotac_link', 'rh_mf_biotac_link', 'rh_rf_biotac_link', 'rh_lf_biotac_link', 'rh_th_biotac_link']
        tip_points = [[self.fingers.get_current_pose(link).pose.position.x, self.fingers.get_current_pose(link).pose.position.y, self.fingers.get_current_pose(link).pose.position.z] for link in tip_links]
        contact_points = [0, 0, 0, 0, 0]
        for x in range(len(tip_points)):
            if np.linalg.norm(np.cross(tip_points[x] - p1, tip_points[x] - p2))/np.linalg.norm(p2 - p1) <= r:
                contact_points[x] = 1
        contact_points = np.array(contact_points, dtype = np.int8)
        print(contact_points)
        return contact_points
