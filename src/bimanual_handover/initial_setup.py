#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.srv import InitialSetupSrv
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from geometry_msgs.msg import PoseStamped

global head, left_arm, tf_buffer, right_arm, hand, gripper, torso

def reset_right_arm():
    '''
    Move the right arm into default configuration and spread fingers of right hand.
    '''
    global right_arm, hand
    # Values from spread_hand script in tams_motorhand
    joint_values = [-0.07571641056445733, -0.05002482770016008, -0.23131151280059523, 0.007094041984987078, 0.02542655924667998, 0.349065850399, 0.020020591171226638, -0.31699402670752413, 0.028414672906931473, 0.08222580050009387, 0.349065850399, 0.008929532157657268, 0.002924555968379014, 0.010452066813274026, 0.349065850399, -0.11378487918959583, 0.014028701132086206, 0.0011093194398269044, 0.349065850399, -0.4480210297267876, 0.048130900509343995, 0.0018684990049854181, -0.11587408526722277, 0.349065850399]
    hand.set_joint_value_target(joint_values)
    hand.go()
    right_arm.set_named_target("right_arm_to_side")
    right_arm.go()

def grasp_object():
    '''
    Interact with human operator to grasp object initially with the left gripper.
    '''
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

def move_fixed():
    '''
    Move head and torso of the robot into the default configuration for the handover.
    Also move the left arm into the correct pose to initially grasp the object with the gripper.
    '''
    global head, left_arm, torso
    head.set_joint_value_target([0.0, 0.872665])
    head.go()
    torso.set_joint_value_target([0.0167])
    torso.go()
    target_pose = PoseStamped()
    target_pose.header.frame_id = "base_footprint"
    target_pose.pose.position.x = 0.5027
    target_pose.pose.position.y = 0.6274
    target_pose.pose.position.z = 0.6848
    target_pose.pose.orientation.w = 1.0
    left_arm.set_pose_target(target_pose)
    left_arm.go()

def initial_setup(req):
    reset_right_arm()
    if req.mode == "fixed":
        move_fixed()
        grasp_object()
    return True

def shutdown():
    roscpp_shutdown()

def main():
    global head, left_arm, right_arm, hand, gripper, torso
    rospy.init_node('init_gripper_stub')
    rospy.on_shutdown(shutdown)
    left_arm = MoveGroupCommander('left_arm', ns = "/")
    left_arm.set_max_velocity_scaling_factor(1.0)
    left_arm.set_max_acceleration_scaling_factor(1.0)
    gripper = MoveGroupCommander('left_gripper', ns = "/")
    gripper.set_max_velocity_scaling_factor(1.0)
    gripper.set_max_acceleration_scaling_factor(1.0)
    head = MoveGroupCommander('head', ns = "/")
    right_arm = MoveGroupCommander('right_arm_pr2', ns = "/")
    right_arm.set_max_velocity_scaling_factor(1.0)
    right_arm.set_max_acceleration_scaling_factor(1.0)
    hand = MoveGroupCommander('right_hand', ns = "/")
    hand.set_max_velocity_scaling_factor(1.0)
    hand.set_max_acceleration_scaling_factor(1.0)
    torso = MoveGroupCommander('torso', ns = "/")
    torso.set_max_velocity_scaling_factor(1.0)
    torso.set_max_acceleration_scaling_factor(1.0)
    rospy.Service('initial_setup_srv', InitialSetupSrv, initial_setup)
    rospy.spin()

if __name__ == "__main__":
    main()
