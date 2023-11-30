#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.src import ProcessPC
from moveit_commander import MoveGroupCommander
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from copy import deepcopy
import math
from tf.transformations import quaternion_from_euler, quternion_multiply
from geometry_msg.msg import Quaternion

global current_pc

def pc_received(pc):
    global current_pc
    current_pc = pc

def main():
    global current_pc
    current_pc = None
    process_pc_srv = rospy.ServiceProxy('process_pc_srv', ProcessPC)
    pc_sub = rospy.Subscriber('pc/pc_filtered', PointCloud2, pc_received)
    z_rots = 4
    # Determine rotation angle based on number of rotations
    rot_angle = math.pi/(r_rots)
    left_arm = MoveGroupCommander("left_arm", ns = "/")
    initial_joint_values = left_arm.get_current_joint_values()
    pcs = []
    for i in range(z_rots):
        result = process_pc_srv(True)
        if not result.success:
            rospy.logerr('Pc processing was unsuccessful. Terminating pc collection.')
            return False
        pcs.append(deepcopy(current_pc))
        current_pose = left_arm.get_current_pose()
        rotation = quaternion_from_euler(0, 0, rot_angle)
        current_pose.pose.orientation = Quaternion(*quaternion_multiply(current_pose.pose.orientation, rot_angle))
        left_arm.set_pose_target(current_pose)
        left_arm.go()
    left_arm.set_joint_value_target(initial_joint_values)
    left_arm.go()

if __name__ == "__main__":
    main()
