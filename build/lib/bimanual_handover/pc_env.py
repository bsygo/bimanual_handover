#!/usr/bin/env python3

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from copy import deepcopy
import gym
import numpy as np
import math
from gym import spaces
import bimanual_handover.syn_grasp_gen as sgg
import sensor_msgs.point_cloud2 as pc2

class SimpleEnv(gym.Env):

    def __init__(self, fingers, pc, ps = None):
        super().__init__()
        self.action_space = spaces.Box(low = np.array([-1.5, -0.65, -1.25]), high = np.array([1.5, 0.65, 1.25]), dtype = np.float64) # + decision if finished, limits found through manual testing
        self.observation_space = spaces.Dict({"finger_joints": spaces.Box(-1.134, 1.658, shape = (22,), dtype = np.float64), "finger_contacts": spaces.MultiBinary(5)})# (last_action, finger_joints, finger_contacts) {"last_action": spaces.Box(low = np.array([-1.5, -0.65, -1.25]), high = np.array([1.5, 0.65, 1.25]), dtype = np.float64), 
        self.fingers = fingers
        self.pc = pc
        self.ps = ps
        self.pca_con = sgg.SynGraspGen() 
        self.last_joints = None
        self.debug_pub = rospy.Publisher('env_debug', PoseStamped)
        self.log_file = open(rospkg.RosPack().get_path('bimanual_handover') + "/logs/log.txt", 'w')

    def step(self, action):
        reward = 0
        terminated = False
        try:
            self.pca_con.move_joint_config(action)
        except Exception as e:
            reward = -1.0
        contact_points = self.__contact_points()
        new_joints = np.array(self.fingers.get_current_joint_values())
        observation = {}
#        observation["last_action"] = action
        observation["finger_joints"] = new_joints
        observation["finger_contacts"] = contact_points
        if np.sum(contact_points) > 0 and reward != -1:
            reward = np.sum(contact_points, dtype = np.float64)
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
        self.last_joints = observation["finger_joints"]
        truncated = False
        info = {}
        print(action)
        print(reward)
        self.log_file.write("{}, {} \n".format(action, reward))
        return observation, reward, terminated, info

    def reset(self):
        self.fingers.set_named_target('open')
        self.fingers.go()
        self.pca_con.set_initial_hand_joints()
        contact_points = self.__contact_points()
        observation = {}
#        observation["last_action"] = np.array([0, 0, 0], dtype = np.float64)
        observation["finger_joints"] = np.array(self.fingers.get_current_joint_values())
        observation["finger_contacts"] = contact_points
        self.last_joints = observation["finger_joints"]
        info = {}
        return observation#, info

    def render(self):
        return

    def close(self):
        self.log_file.close()
        return

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
        print(r)
        for x in range(len(tip_points)):
            if np.linalg.norm(np.cross(tip_points[x] - p1, tip_points[x] - p2))/np.linalg.norm(p2 - p1) <= r:
                contact_points[x] = 1
        contact_points = np.array(contact_points, dtype = np.int8)
        print(contact_points)
        return contact_points

