#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from sr_robot_msgs.msg import BiotacAll, Biotac
from geometry_msgs.msg import WrenchStamped, Wrench, Vector3

def main():
    rospy.init_node("test_publisher")
    joint_state_pub = rospy.Publisher("/hand/joint_states", JointState, queue_size = 1)
    force_pub = rospy.Publisher("/ft/l_gripper_motor", WrenchStamped, queue_size = 1)
    biotac_pub = rospy.Publisher("/hand/rh/tactile", BiotacAll, queue_size = 1)

    joint_state_msg = JointState()
    joint_state_msg.name = ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1', 'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1', 'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1', 'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1', 'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1']
    joint_state_msg.position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    joint_state_msg.effort = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    force_msg = WrenchStamped()
    wrench_msg = Wrench()
    wrench_msg.force = Vector3(0, 0, 0)
    wrench_msg.torque = Vector3(0, 0, 0)
    force_msg.wrench = wrench_msg

    biotac_msg = BiotacAll()
    tactiles = []
    for i in range(5):
        biotac = Biotac()
        tactiles.append(biotac)
    biotac_msg.tactiles = tactiles
    while not rospy.is_shutdown():
        joint_state_pub.publish(joint_state_msg)
        force_pub.publish(force_msg)
        biotac_pub.publish(biotac_msg)

if __name__ == "__main__":
    main()
