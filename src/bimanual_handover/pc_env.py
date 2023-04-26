#!/usr/bin/env python3

import gym
import numpy as np
import math
from gym import spaces
import bimanual_handover.syn_grasp_gen as sgg
import sensor_msgs.point_cloud2 as pc2

class SimpleEnv(gym.Env):

    def __init__(self, fingers, pc, ps = None):
        super().__init__()
        self.action_space = spaces.Box(low = -2.0, high = 2.0, shape = (3,), dtype = np.float64) # + decision if finished
        self.observation_space = spaces.Dict({"last_action": spaces.Box(-2.0, 2.0, shape = (3,), dtype = np.float64), "finger_joints": spaces.Box(-1.134, 1.658, shape = (22,), dtype = np.float64), "finger_contacts": spaces.MultiBinary(5)})# (last_action, finger_joints, finger_contacts) 
        self.fingers = fingers
        self.pc = pc
        self.ps = ps
        self.pca_con = sgg.SynGraspGen() 
        self.last_joints = None

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
        observation["last_action"] = action
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
        print(reward)
        return observation, reward, terminated, info

    def reset(self):
        self.fingers.set_named_target('open')
        self.fingers.go()
        self.pca_con.set_initial_hand_joints()
        contact_points = self.__contact_points()
        observation = {}
        observation["last_action"] = np.array([0, 0, 0], dtype = np.float64)
        observation["finger_joints"] = np.array(self.fingers.get_current_joint_values())
        observation["finger_contacts"] = contact_points
        self.last_joints = observation["finger_joints"]
        info = {}
        return observation#, info

    def render(self):
        return

    def close(self):
        return

    def __contact_points(self):
        cylinder = self.ps.get_objects(['can'])['can']
        h = cylinder.primitives[0].dimensions[0]
        r = cylinder.primitives[0].dimensions[1]
        pose = cylinder.pose
        p1 = pose.position
        p1.z += h/2
        p1 = np.array([p1.x, p1.y, p1.z])
        p2 = pose.position
        p2.z += -h/2
        p2 = np.array([p2.x, p2.y, p2.z])
        tip_links = ['rh_ff_biotac_link', 'rh_mf_biotac_link', 'rh_rf_biotac_link', 'rh_lf_biotac_link', 'rh_th_biotac_link']
        tip_points = [[self.fingers.get_current_pose(link).pose.position.x, self.fingers.get_current_pose(link).pose.position.y, self.fingers.get_current_pose(link).pose.position.z] for link in tip_links]
        contact_points = [0, 0, 0, 0, 0]
        print(r)
        for x in range(len(tip_points)):
            print(np.abs(np.linalg.norm(np.cross(p1 - tip_points[x], p2 - tip_points[x])))/np.linalg.norm(p1 - tip_points[x]))
            if np.abs(np.linalg.norm(np.cross(p1 - tip_points[x], p2 - tip_points[x])))/np.linalg.norm(p1 - tip_points[x]) <= r:
                contact_points[x] = 1
        '''
        points = pc2.read_points(self.pc, field_names = ("x", "y", "z"), skip_nans = True)
        for point in points:
            for x in range(len(tip_points)):
                if math.dist(point, list(tip_points[x])) <= 0.01:
                    contact_points[x] = 1
        '''
        contact_points = np.array(contact_points, dtype = np.int8)
        print(contact_points)
        return contact_points

