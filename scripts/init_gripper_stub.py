#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.srv import InitGripper
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown

def init_gripper(req):
    roscpp_initialize("")
    left_arm = MoveGroupCommander('left_arm', ns = "/")
    head = MoveGroupCommander('head', ns = "/")
    head.set_joint_value_target([0.0, 0.872665])
    head.go()
    joint_values = dict(l_shoulder_pan_joint=0.6877300386981536, l_shoulder_lift_joint=0.0014527860014343034, l_upper_arm_roll_joint=1.988643872487468, l_forearm_roll_joint=-0.48605351559908117, l_elbow_flex_joint=-1.7236114019354039, l_wrist_flex_joint=-0.6663365621588351, l_wrist_roll_joint=-0.9874690540253139)
    left_arm.set_joint_value_target(joint_values)
    left_arm.go()
    return True

def shutdown():
    roscpp_shutdown()

def main():
    rospy.init_node('init_gripper_stub')
    rospy.on_shutdown(shutdown)
    rospy.Service('init_gripper_srv', InitGripper, init_gripper)
    rospy.spin()

if __name__ == "__main__":
    main()
