#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
from sensoe_msgs.msg import JointState
from stable_baselines3 import PPO
import bimanual_handover.syn_grasp_gen as sgg
from bimanual_handover_msgs.srv import ModelMoverSrv

class ModelMover():

    def __init__(self, model_name = "ppo_model_08_09_2023_16_41.zip"):
        rospy.init_node('model_mover')
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('bimanual_handover')
        model_path = pkg_path + "/models/" + model_name
        self.model = PPO.load(model_path)
        self.current_obs = dict(joints = np.zeros((24,), np.float32))
        self.pca_con = sgg.SynGraspGen()
        self.joint_state_sub = rospy.Subscriber('/hand/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        rospy.Service('model_mover_srv', ModelMoverSrv, self.close_hand)
        rospy.spin()

    def joint_callback(self, joint_state):
        indices = [joint_state.name.index(joint_name) for joint_name in self.joints]
        current_joint_values = [joint_state.position[x] for x in indices]
        current_joint_values = np.array(current_joint_values, dtype = np.float32)
        self.current_obs["joints"] = current_joint_values

    def get_next_action(self):
        action, _states = self.model.predict(self.current_obs)
        return action

    def exec_action(self, action):
        self.pca_con.move_joint_config(action[:5])

    def close_hand(self, req):
        for i in range(5):
            action = get_next_action()
            exec_action(action)
        return True

if __name__ == "__main__":
    mm = ModelMover()
