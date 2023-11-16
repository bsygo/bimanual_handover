#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.srv import InitGripper
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from geometry_msgs.msg import PoseStamped

global head, left_arm, tf_buffer, debug_pub

def move_pose(target_pose):
    global head, left_arm
    left_arm.set_pose_target(target_pose)
    left_arm.go()
    # Call look at service for head
    return

def move_head_only():
    global head
    head.set_joint_value_target([0.0, 0.872665])
    head.go()

def move_fixed():
    global head, left_arm
    head.set_joint_value_target([0.0, 0.872665])
    head.go()
    joint_values = dict(l_shoulder_pan_joint=0.6877300386981536, l_shoulder_lift_joint=0.0014527860014343034, l_upper_arm_roll_joint=1.988643872487468, l_forearm_roll_joint=-0.48605351559908117, l_elbow_flex_joint=-1.7236114019354039, l_wrist_flex_joint=-0.6663365621588351, l_wrist_roll_joint=-0.9874690540253139)
    left_arm.set_joint_value_target(joint_values)
    left_arm.go()

def init_gripper(req):
    if req.mode == "fixed":
        move_fixed()
    elif req.mode == "pose":
        move_pose(req.target_pose)
    elif req.mode == "head":
        move_head_only()
    return True

def shutdown():
    roscpp_shutdown()

def main():
    global head, left_arm
    rospy.init_node('init_gripper_stub')
    rospy.on_shutdown(shutdown)
    roscpp_initialize("")
    left_arm = MoveGroupCommander('left_arm', ns = "/")
    head = MoveGroupCommander('head', ns = "/")
    debug_pub = rospy.Publisher('debug_head_rot', PoseStamped, queue_size = 1, latch = True)
    rospy.Service('init_gripper_srv', InitGripper, init_gripper)
    rospy.spin()

if __name__ == "__main__":
    main()
