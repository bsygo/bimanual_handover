#!/usr/bin/env python3

import rospy
import sys
from moveit_commander import MoveGroupCommander, RobotCommander
import hand_synergy.hand_synergy as hs
import numpy as np
from moveit_msgs.msg import DisplayRobotState, DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

class SynGraspGen():

    def __init__(self, display_state = False):
        self.synergy = hs.HandSynergy(n_components = 3)
        self.hand = MoveGroupCommander("right_hand", ns = "/")
        self.robot = RobotCommander()
        self.display_state = display_state
        self.display_state_pub = rospy.Publisher("synergies_debug", DisplayRobotState, latch = True, queue_size = 1)
        self.display_trajectory_pub = rospy.Publisher("synergies_traj_debug", DisplayTrajectory, latch = True, queue_size = 1)
        self.joint_order = self.hand.get_active_joints()

    def gen_joint_config(self, alphas, normalize = False):
        current_joint_config = self.hand.get_current_joint_values()
        joint_names = ['WRJ2', 'WRJ1'] + hs.FINGER_JOINTS_ORDER
        order = []
        for name in joint_names:
            order.append(self.joint_order.index('rh_' + name))
        switched_joint_config = np.array(current_joint_config)[order]
        generated_joint_config = self.synergy.get_shadow_target_joints(switched_joint_config, np.array([0, 0]), alphas)
        joint_dict = {}
        for i in range(len(joint_names)):
            joint_dict.update({'rh_' + joint_names[i] : generated_joint_config[i]})

        # change THJ4 and THJ5
        temp = joint_dict['rh_THJ4']
        joint_dict['rh_THJ4'] = joint_dict['rh_THJ5']
        joint_dict['rh_THJ5'] = temp

        if normalize:
            joint_dict = self.normalize(joint_dict, current_joint_config)
        joint_dict = self.enforce_bounds(joint_dict)
        joint_dict = self.limit_joints(joint_dict, current_joint_config)

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

    def get_pca_config(self, joint_values = None):
        if joint_values is None:
            joint_config = self.hand.get_current_joint_values()
        else:
            joint_config = joint_values
        joint_names = ['WRJ2', 'WRJ1'] + hs.FINGER_JOINTS_ORDER
        order = []
        for name in joint_names:
            order.append(self.joint_order.index('rh_' + name))
        switched_joint_config = np.array(joint_config)[order]
        pca_values = self.synergy.pca.transform(switched_joint_config[2:].reshape(1, -1))
        return pca_values

    def normalize(self, joint_dict, current_joint_config):
        joint_diffs = {}
        for key, value in joint_dict.items():
            joint_diffs[key] = joint_dict[key] - current_joint_config[self.joint_order.index(key)]
        max_diff = abs(joint_diffs[max(joint_diffs, key = lambda y: abs(joint_diffs[y]))])
        if max_diff > 0.15708:
            for key, value in joint_diffs.items():
                joint_diffs[key] = value/max_diff * 0.15708
        new_joint_config = {}
        for key, value in joint_diffs.items():
            new_joint_config[key] = current_joint_config[self.joint_order.index(key)] + joint_diffs[key]
        return new_joint_config

    def limit_joints(self, joint_dict, current_joint_config):
        limit_joints = ['rh_FFJ4', 'rh_MFJ4', 'rh_RFJ4', 'rh_LFJ4']
        for joint in limit_joints:
            joint_dict[joint] = current_joint_config[self.joint_order.index(joint)] 
        return joint_dict

    def enforce_bounds(self, joint_dict):
        for joint in joint_dict:
            joint_object = self.robot.get_joint(joint)
            bounds = joint_object.bounds()
            if joint_dict[joint] < bounds[0]:
                joint_dict[joint] = bounds[0]
            elif joint_dict[joint] > bounds[1]:
                joint_dict[joint] = bounds[1]
        return joint_dict

    def exec_joint_config(self, joint_dict):
        self.hand.set_joint_value_target(joint_dict)
        result = self.hand.go()
        return result

    def move_joint_config(self, alphas = np.array([0, 0, 0])):
        joint_dict = self.gen_joint_config(alphas)
        self.publish_trajectory(joint_dict)
        result = self.exec_joint_config(joint_dict)
        return result

    def publish_trajectory(self, joint_dict):
        msg = DisplayTrajectory()
        msg.trajectory = [RobotTrajectory()]
        msg.trajectory[0].joint_trajectory.points = [JointTrajectoryPoint()]
        for key in joint_dict:
            msg.trajectory[0].joint_trajectory.joint_names.append(key)
            msg.trajectory[0].joint_trajectory.points[0].positions.append(joint_dict[key])
        msg.trajectory_start = self.robot.get_current_state()
        self.display_trajectory_pub.publish(msg)

if __name__ == "__main__":
    syn_grasp_gen = SynGraspGen()
