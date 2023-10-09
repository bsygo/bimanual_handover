#!/usr/bin/env python3

import rospy
import rospkg
import math
import numpy as np
from sensor_msgs.msg import JointState
from stable_baselines3 import PPO
import bimanual_handover.syn_grasp_gen as sgg
from bimanual_handover_msgs.srv import ModelMoverSrv

class ModelMover():

    def __init__(self, model_name = "ppo_model_06_10_2023_13_11.zip", model_type = "pca"):
        rospy.init_node('model_mover')
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('bimanual_handover')
        model_path = pkg_path + "/models/" + model_name
        self.model = PPO.load(model_path)
        self.model_type = model_type
        if model_type == "joints":
            self.current_obs = dict(joints = np.zeros((24,), np.float32))
        elif model_type == "pca":
            self.current_obs = np.zeros((3,), np.float32)
        else:
            rospy.logerr("Unknown model_type {}.".format(self.model_type))
        self.joints = ['rh_WRJ2', 'rh_WRJ1', 'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1', 'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1', 'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1', 'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1', 'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1']
        self.pca_con = sgg.SynGraspGen()
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        #self.joint_state_sub = rospy.Subscriber('/hand/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        rospy.Service('model_mover_srv', ModelMoverSrv, self.close_hand)
        rospy.spin()

    def joint_callback(self, joint_state):
        if not self.joints[0] in joint_state.name:
            return
        indices = [joint_state.name.index(joint_name) for joint_name in self.joints]
        current_joint_values = [joint_state.position[x] for x in indices]
        current_joint_values = np.array(current_joint_values, dtype = np.float32)
        if self.model_type == "joints":
            self.current_obs["joints"] = current_joint_values
        elif self.model_type == "pca":
            self.current_obs = np.array(self.pca_con.get_pca_config(current_joint_values)[0][:3], dtype = np.float32)

    def get_next_action(self):
        action, _states = self.model.predict(self.current_obs)
        return action

    def exec_action(self, action):
        if self.model_type == "joints":
            joint_dict = self.pca_con.gen_joint_config(action[:3])
            joint_list = [joint_dict[joint] for joint in self.joints]
            joint_arr = np.array(joint_list, np.float32)
            print(joint_arr)
            print(self.current_obs['joints'])
            print(-math.dist(self.current_obs['joints'], joint_list))
            self.pca_con.move_joint_config(action[:3])
        elif self.model_type == "pca":
            self.pca_con.move_joint_config(action)

    def close_hand(self, req):
        for i in range(5):
            action = self.get_next_action()
            self.exec_action(action)
        return True

if __name__ == "__main__":
    mm = ModelMover()
