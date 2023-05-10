#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2

global publish, pc

def change_publish(value):
    global publish, pc
    if not publish and value:
        publish = value
        pc = rospy.wait_for_message('/azure_kinect/points2', PointCloud2)

def republish_pc():
    global pc
    pub = rospy.Publisher('/cloud_pcd', PointCloud2)
    pc = rospy.wait_for_message('/azure_kinect/points2', PointCloud2)
    while not rospy.is_shutdown():
        if publish:
            pc.header.stamp = rospy.Time(0)
            pub.publish(pc)
        rospy.sleep(1)

def main():
    global publish, pc
    rospy.init_node('capture_pc')
    publish = False
    rospy.Subscriber('change_publish', bool, change_publish, queue_size = 1)
    republish_pc()

if __name__ == "__main__":
    main()
