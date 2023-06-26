#!/usr/bin/env python3

import rospy
import sys
from moveit_commander import MoveGroupCommander, RobotCommander
import hand_synergy.hand_synergy as hs
import numpy as np
from moveit_msgs.msg import DisplayRobotState

class SynGraspGen():

    def __init__(self, display_state = False):
        # self.zero_pca_action = np.array([-1.5, -0.55, 0.25, 0.15, 0.05]) # offsets to center the 0 pose
        self.synergy = hs.HandSynergy(n_components = 5)
        self.hand = MoveGroupCommander("right_hand")
        self.display_state = display_state
        # self.initial_hand_joints = self.hand.get_current_joint_values()
        self.display_state_pub = rospy.Publisher("synergies_debug", DisplayRobotState, latch = True, queue_size = 1)

    # def set_initial_hand_joints(self):
    #   self.inital_hand_joints = self.hand.get_current_joint_values() 

    def gen_joint_config(self, alphas):
        #sh_joints = self.initial_hand_joints
        sh_joints = self.hand.get_current_joint_values()
        sh_names = self.hand.get_active_joints()
        joint_names = ['WRJ2', 'WRJ1'] + hs.FINGER_JOINTS_ORDER
        order = []
        for name in joint_names:
            order.append(sh_names.index('rh_' + name))
        switched_sh_joints = np.array(sh_joints)[order]
        joints = self.synergy.get_shadow_target_joints(switched_sh_joints, np.array([0, 0]), alphas)# + self.zero_pca_action)
        joint_dict = {}
        for i in range(len(joint_names)):
            joint_dict.update({'rh_' + joint_names[i] : joints[i]})

        # change THJ4 and THJ5
        temp = joint_dict['rh_THJ4']
        joint_dict['rh_THJ4'] = joint_dict['rh_THJ5']
        joint_dict['rh_THJ5'] = temp

        if self.display_state:
            robot = RobotCommander()
            robot_state = robot.get_current_state()
            joint_state = robot_state.joint_state
            position = list(joint_state.position)
            for joint in joint_dict:
                index = joint_state.name.index(joint)
                position[index] = joint_dict[joint]
            joint_state.position = tuple(position)
            robot_state.joint_state = joint_state
            display_state = DisplayRobotState()
            display_state.state = robot_state
            self.display_state_pub.publish(display_state)

        return joint_dict

    def exec_joint_config(self, joint_dict):
        self.hand.set_joint_value_target(joint_dict)
        result = self.hand.go()
        return result

    def move_joint_config(self, alphas = np.array([0, 0, 0, 0, 0])):
        joint_dict = self.gen_joint_config(alphas)
        result = self.exec_joint_config(joint_dict)
        return result

if __name__ == "__main__":
    syn_grasp_gen = SynGraspGen()
    syn_grasp_gen.move_joint_config()
