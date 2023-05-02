#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2

global publish

def republish_pc(pc):
    pub = rospy.Publisher('/cloud_pcd', PointCloud2)
    pc = rospy.wait_for_message('/azure_kinect/points2', PointCloud2)
    while publish:
        pc.header.stamp = rospy.Time(0)
        pub.publish(pc)
        rospy.sleep(1)

def main():
    global publish
    rospy.init_node('pc_capture')
    publish = True
    republish_pc(pc)

if __name__ == "__main__":
    main()
