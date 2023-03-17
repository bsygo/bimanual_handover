#!/usr/bin/env python3

import rospy
import moveit_commander
import numpy as np
from gpd_grasp_msgs.msg import GraspConfig
from gpd_grasp_msgs.msg import GraspConfigList
from geometry_msgs.msg import PoseStamped
from pyquaternion import Quaternion

def execute_grasp(grasp_list):
    gripper_pose = PoseStamped()
    gripper_pose.header.frame_id = grasp_list.header
    single_grasp = grasp_list[0]
    gripper_pose.pose.position = single_grasp.bottom
    gripper_pose.pose.orientation = Quaternion(matrix=np.vstack([single_grasp.approach, singel_grasp.binormal, singel_grasp.axis]).T)
    group.set_pose_target(gripper_pose)
    group.go()
    gripper.set_named_target("closed")
    gripper.go()
    group.set_named_target("left_arm_to_side")
    group.go()

if __name__ == '__main__':
    rospy.init_node('pointnetgpd_test')
    group = moveit_commander.MoveGroupCommander("left_arm")
    gripper = moveit_commander.MoveGroupCommander("left_gripper")
    grasp_list = rospy.wait_for_message("/detect_grasps/clustered_grasps", GraspConfigList)
    execute_grasp(grasp_list)
