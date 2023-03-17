#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

def point_transform(points):
    point_pub.publish(points)

def image_transform(image):
    image_pub.publish(image)

def cam_info_transform(cam_info):
    cam_info_pub.publish(cam_info)

if __name__ == '__main__':
    rospy.init_node('bag_transform')
    point_pub = rospy.Publisher('/table_top_points', PointCloud2, queue_size = 10)
    image_pub = rospy.Publisher('azure_kinect/rgb/image_raw', Image, queue_size = 10)
    cam_info_pub = rospy.Publisher('azure_kinect/rgb/camera_info', CameraInfo, queue_size = 10)
    point_sub = rospy.Subscriber('/points2', PointCloud2, point_transform)
    image_sub = rospy.Subscriber('/rgb/image_raw', Image, image_transform)
    cam_info_sub = rospy.Subscriber('/rgb/camera_info', CameraInfo, cam_info_transform)
    while not rospy.is_shutdown():
        rospy.spin()
