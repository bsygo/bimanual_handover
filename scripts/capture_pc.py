#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
import sys

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
    if len(sys.argv) < 2:
        rospy.logerr("Missing argument for capture_pc.")
        return
    elif sys.argv[1]:
        input_topic = 'pc/cloud_pcd'
    else:
        input_topic = '/azure_kinect/points2'
    publish = False
    rospy.Subscriber(input_topic, PointCloud2, save_pc, queue_size = 1)
    rospy.Subscriber('publish_pc', Bool, publish_pc, queue_size = 1)
    pub = rospy.Publisher('pc/pc_raw', PointCloud2, queue_size=5)
    rospy.spin()

if __name__ == "__main__":
    main()
