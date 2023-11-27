#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.srv import InitialSetupSrv
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from geometry_msgs.msg import PoseStamped

global head, left_arm, tf_buffer, right_arm, hand, gripper

def reset_right_arm():
    global right_arm, hand
    # Values from spread_hand script in tams_motorhand
    joint_values = [-0.07571641056445733, -0.05002482770016008, -0.23131151280059523, 0.007094041984987078, 0.02542655924667998, 0.349065850399, 0.020020591171226638, -0.31699402670752413, 0.028414672906931473, 0.08222580050009387, 0.349065850399, 0.008929532157657268, 0.002924555968379014, 0.010452066813274026, 0.349065850399, -0.11378487918959583, 0.014028701132086206, 0.0011093194398269044, 0.349065850399, -0.4480210297267876, 0.048130900509343995, 0.0018684990049854181, -0.11587408526722277, 0.349065850399]
    hand.set_joint_value_target(joint_values)
    hand.go()
    right_arm.set_named_target("right_arm_to_side")
    right_arm.go()

def grasp_object():
    global gripper
    response = input("Do you want to change the current object? [Y/n]")
    if response in ["n", "N", "no", "No"]:
        print("Keeping the current object. Continuing without changing.")
        return
    input("Please prepare to catch current object in the gripper. Press Enter when ready.")
    gripper.set_named_target('open')
    gripper.go()
    input("Please insert the new object into the gripper. Press Enter when ready.")
    gripper.set_named_target('closed')
    gripper.go()

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

def move_fixed(object_type):
    global head, left_arm
    head.set_joint_value_target([0.0, 0.872665])
    head.go()
    if object_type == "can":
        joint_values = dict(l_shoulder_pan_joint=0.6877300386981536, l_shoulder_lift_joint=0.0014527860014343034, l_upper_arm_roll_joint=1.988643872487468, l_forearm_roll_joint=-0.48605351559908117, l_elbow_flex_joint=-1.7236114019354039, l_wrist_flex_joint=-0.6663365621588351, l_wrist_roll_joint=-0.9874690540253139)
    elif object_type == "book":
        joint_values = [0.497912812475168, 0.6417141983535197, 1.5318181439664142, -1.894157976916556, -2.7289803579735015, -0.8217423864483477, 1.8652765105646032]
    left_arm.set_joint_value_target(joint_values)
    left_arm.go()

def initial_setup(req):
    reset_right_arm()
    if req.mode == "fixed":
        move_fixed(req.object_type)
        grasp_object()    
    elif req.mode == "pose":
        move_pose(req.target_pose)
        grasp_object()    
    elif req.mode == "head":
        move_head_only()
    return True

def shutdown():
    roscpp_shutdown()

def main():
    global head, left_arm, right_arm, hand, gripper
    rospy.init_node('init_gripper_stub')
    rospy.on_shutdown(shutdown)
    left_arm = MoveGroupCommander('left_arm', ns = "/")
    gripper = MoveGroupCommander('left_gripper', ns = "/")
    head = MoveGroupCommander('head', ns = "/")
    right_arm = MoveGroupCommander('right_arm_pr2', ns = "/")
    hand = MoveGroupCommander('right_hand', ns = "/")
    rospy.Service('initial_setup_srv', InitialSetupSrv, initial_setup)
    rospy.spin()

if __name__ == "__main__":
    main()
