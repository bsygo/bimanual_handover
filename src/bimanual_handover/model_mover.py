#!/usr/bin/env python3

import rospy
import rospkg
import math
import numpy as np
from sensor_msgs.msg import JointState
from sr_robot_msgs.msg import BiotacAll
from stable_baselines3 import PPO, SAC
import bimanual_handover.syn_grasp_gen as sgg
from bimanual_handover_msgs.srv import ModelMoverSrv
from moveit_commander import MoveGroupCommander
from copy import deepcopy

class ModelMover():

    def __init__(self, model_name = "sac_checkpoint_23_10_2023_15_47_12900_steps.zip"):
        rospy.init_node('model_mover')
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('bimanual_handover')
        model_path = pkg_path + "/models/" + "checkpoints/" + model_name
        self.model = SAC.load(model_path)
        self.model.load_replay_buffer(pkg_path + "/models/checkpoints/sac_checkpoint_23_10_2023_15_47_replay_buffer_12900_steps")
        self.fingers = MoveGroupCommander('right_fingers', ns="/")
        self.fingers.set_max_velocity_scaling_factor(1.0)
        self.fingers.set_max_acceleration_scaling_factor(1.0)
        self.initial_tactile = None
        self.current_tactile = None
        self.current_pca = None
        self.tactile_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, self.tactile_callback)
        self.joints = ['rh_WRJ2', 'rh_WRJ1', 'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1', 'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1', 'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1', 'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1', 'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1']
        self.pca_con = sgg.SynGraspGen()
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        #self.joint_state_sub = rospy.Subscriber('/hand/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        rospy.Service('model_mover_srv', ModelMoverSrv, self.close_hand)
        rospy.spin()

    def tactile_callback(self, tactile):
        self.current_tactile = [x.pdc for x in tactile.tactiles]

    def joint_callback(self, joint_state):
        if not self.joints[0] in joint_state.name:
            return
        indices = [joint_state.name.index(joint_name) for joint_name in self.joints]
        current_joint_values = [joint_state.position[x] for x in indices]
        current_joint_values = np.array(current_joint_values, dtype = np.float32)
        self.current_pca = self.pca_con.get_pca_config(current_joint_values)[0][:3]

    def get_next_action(self):
        current_biotac = deepcopy(self.current_tactile)
        current_observation = [current_biotac[x] - self.initial_tactile[x] for x in range(len(current_biotac))]
        current_pca = self.pca_con.get_pca_config()[0][:3]
        for pca in current_pca:
            current_observation.append(pca)
        observation = np.array(current_observation, dtype = np.float32)
        action, _states = self.model.predict(observation, deterministic = True)
        return action

    def exec_action(self, action):
        # Stop moving if enough contacts have been made
        current_biotac = deepcopy(self.current_tactile)
        current_biotac_diff = [current_biotac[x] - self.initial_tactile[x] for x in range(len(current_biotac))]
        contacts = [True if diff >=20 else False for diff in current_biotac_diff]
        # Reduce normalized actions
        action[0] = (action[0] - 1)/2
        action[1] = (action[1] - 1)/2
        # Get new config
        result = self.pca_con.gen_joint_config(action[:3], normalize = True)
        # Remove wrist joints
        del result['rh_WRJ1']
        del result['rh_WRJ2']

        # Remove fingers with contact (optional, testing) -> Mabe switch to not close anymore instead of no movement at all
        del_keys = []
        if contacts[0]:
            for key in result.keys():
                if 'rh_FFJ' in key:
                    del_keys.append(key)
        if contacts[1]:
            for key in result.keys():
                if 'rh_MFJ' in key:
                    del_keys.append(key)
        if contacts[2]:
            for key in result.keys():
                if 'rh_RFJ' in key:
                    del_keys.append(key)
        if contacts[3]:
            for key in result.keys():
                if 'rh_LFJ' in key:
                    del_keys.append(key)
        if contacts[4]:
            for key in result.keys():
                if 'rh_THJ' in key:
                    del_keys.append(key)
        for key in del_keys:
            del result[key]

        self.fingers.set_joint_value_target(result)
        self.fingers.go()

    def close_hand(self, req):
        self.initial_tactile = deepcopy(self.current_tactile)
        for i in range(20):
            action = self.get_next_action()
            self.exec_action(action)
        return True

if __name__ == "__main__":
    mm = ModelMover()
