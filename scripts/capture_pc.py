#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool

global pub, publish

def save_pc(new_pc):
    global pub, publish
    if publish:
        pub.publish(new_pc)
        publish = False

def publish_pc(publish_pc):
    global publish
    publish = publish_pc

def main():
    global pub, publish
    rospy.init_node('capture_pc')
    publish = False
    pc = rospy.wait_for_message('/azure_kinect/points2', PointCloud2)
    rospy.Subscriber('/azure_kinect/points2', PointCloud2, save_pc, queue_size = 1)
    rospy.Subscriber('handover/pc/publish_pc', Bool, publish_pc, queue_size = 1)
    pub = rospy.Publisher('/cloud_pcd', PointCloud2, queue_size=5)
    rospy.spin()

if __name__ == "__main__":
    main()
