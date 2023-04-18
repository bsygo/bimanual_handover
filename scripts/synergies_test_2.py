#!/usr/bin/env python3

import rospy 
from moveit_commander import RobotCommander, roscpp_initialize, MoveGroupCommander, roscpp_shutdown()
import hand_synergy.hand_synergy as hs
import numpy as np
from moveit_msgs.msg import DisplayRobotState

def end_moveit():
    roscpp_shutdown()

def main():
    rospy.init_node('synergies_test')
    roscpp_initialize("")
    rospy.on_shutdown(end_moveit)
    robot = RobotCommander()
    hand = MoveGroupCommander("right_hand")
    display_state_pub = rospy.Publisher("synergies_debug", DisplayRobotState, latch = True, queue_size = 1)
    '''
    data = hs.get_pca_grasp_data(grasp_type_value = 0)
    pca, range_min, range_max = hs.fit_pca(data, 3)

    alphas = np.array([0, 0, 0])
    joints = pca.mean_ + np.dot(alphas, pca.components_)

    current_pca = pca.transform(np.zeros([1, 22]))
    joints = np.append(joints, [0, 0])
    active_joints = hand.get_active_joints()
    joint_dict = {}
    joint_names = hs.FINGER_JOINTS_ORDER + ['WRJ2', 'WRJ1']
    for joint in active_joints:
        index = joint_names.index(joint[3:])
        joint_dict.update({joint : joints[index]})
    
    unused_joints = ['rh_FFJ1', 'rh_MFJ1', 'rh_RFJ1', 'rh_LFJ1', 'rh_THJ1']
    for joint in unused_joints:
        joint_dict[joint] = 0.349066
    '''
    joint_names = ['WRJ2', 'WRJ1'] + hs.FINGER_JOINTS_ORDER

    synergy = hs.HandSynergy()
    sh_joints = hand.get_current_joint_values()
    sh_names = hand.get_active_joints()
    assert(len(sh_names) == len(sh_joints))
    switched_sh_joints = []
    for name in joint_names:
        index = sh_names.index('rh_' + name)
        switched_sh_joints.append(sh_joints[index])
    switched_sh_joints = np.array(switched_sh_joints)
    joints = synergy.get_shadow_target_joints(switched_sh_joints, np.array([0, 0]), np.array([-2, 0, 0]))
    print(joints)
    joint_dict = {}
    for i in range(len(joint_names)):
        joint_dict.update({'rh_' + joint_names[i] : joints[i]})
    print(joint_dict)
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
    display_state_pub.publish(display_state)
    
    print("fin")
    rospy.spin()
#    hand.set_joint_value_target(joint_dict)
#    hand.go()

if __name__ == "__main__":
    main()
    rospy.spin()
