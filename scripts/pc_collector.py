#!/usr/bin/env python3

import rospy
from moveit_commander import MoveGroupCommander
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool

def main():
    capture_pub = rospy.Publisher('publish_pc', Bool, queue_size = 1)
    pc_sub = rospy.Subscriber('pc/pc_raw', PointCloud2)
    z_rots = 3
    left_arm = MoveGroupCommander("left_arm", ns = "/")
    initial_joint_values = left_arm.get_current_joint_values()
    pcs = []
    for i in range(z_rots):

if __name__ == "__main__":
    main()
