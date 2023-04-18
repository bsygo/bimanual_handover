#!/usr/bin/env python3

import rospy
from moveit_commander import RobotCommander, roscpp_initialize, MoveGroupCommander, roscpp_shutdown()
import tams_hand_synergies
import rospkg
import random

def end_moveit():
    roscpp_shutdown()

if __name__ == "__main__":
    rospy.init_node('synergies_test')
    roscpp_initialize("")
    roscpp_shutdown(end_moveit)
    robot = RobotCommander()
    hand = MoveGroupCommander("right_hand")
    hand_synergies = tams_hand_synergies.HandSynergies()
    coefficients = (0.6, 0, 0.32, 0, 0, 0, 0, 0, 0)
    joint_positions = hand_synergies.compute(coefficients)
    joint_dict = {}
    rospack = rospkg.RosPack()
    path = rospack.get_path('tams_hand_synergies')
    with open(path + '/data/synergies.txt') as f:
        line = f.readline()
        read_joints = line.split()
        f.close()
    for i in range(len(read_joints)):
        joint_dict.update({'rh_' + read_joints[i] : joint_positions[i]})
    unused_joints = ['rh_FFJ1', 'rh_MFJ1', 'rh_RFJ1', 'rh_LFJ1', 'rh_THJ1']
    for joint in unused_joints:
        del joint_dict[joint]
    print(joint_dict)
    hand.set_joint_value_target(joint_dict)
    hand.go()


