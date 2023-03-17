#!/usr/bin/env python3

import rospy
import sys
from moveit_commander import RobotCommander, roscpp_initialize, MoveGroupCommander
from sensor_msgs.msg import JointState

global la, sub, ra, torso, head

def check_js(js):
    joints = la.get_active_joints()
    for joint in joints:
        if not joint in js.name:
            return
    set_right_arm(js)
    set_left_arm(js)
    set_torso(js)
    set_head(js)
    rospy.loginfo('Robot positioning finished.')
    sub.unregister()

def set_right_arm(js):
    filtered_js = JointState()
    joints = ra.get_active_joints()
    indices = [js.name.index(joint_name) for joint_name in joints]
    filtered_js.name = joints
    filtered_js.position = [js.position[x] for x in indices]
    ra.set_joint_value_target(filtered_js.position)
    ra.go()

def set_left_arm(js):
    filtered_js = JointState()
    joints = la.get_active_joints()
    indices = [js.name.index(joint_name) for joint_name in joints]
    filtered_js.name = joints
    filtered_js.position = [js.position[x] for x in indices]
    filtered_js.position[5] = -1.01
    la.set_joint_value_target(filtered_js.position)
    la.go()

def set_torso(js):
    torso.set_max_velocity_scaling_factor(0.9)
    torso.set_max_acceleration_scaling_factor(0.9)
    filtered_js = JointState()
    joints = torso.get_active_joints()
    indices = [js.name.index(joint_name) for joint_name in joints]
    filtered_js.name = joints
    filtered_js.position = [js.position[x] for x in indices]
    torso.set_joint_value_target(filtered_js.position)
    torso.go()

def set_head(js):
    filtered_js = JointState()
    joints = head.get_active_joints()
    indices = [js.name.index(joint_name) for joint_name in joints]
    filtered_js.name = joints
    filtered_js.position = [js.position[x] for x in indices]
    head.set_joint_value_target(filtered_js.position)
    head.go()

if __name__ =='__main__':
    global la, sub, ra, torso, head
    rospy.init_node('pc_test_mover')
    roscpp_initialize(sys.argv)
    rc = RobotCommander()
    la = MoveGroupCommander('left_arm')
    ra = MoveGroupCommander('right_arm_pr2')
    torso = MoveGroupCommander('torso')
    head =MoveGroupCommander('head')
    sub = rospy.Subscriber('/pc_joint_states', JointState, check_js)
    rospy.spin()

