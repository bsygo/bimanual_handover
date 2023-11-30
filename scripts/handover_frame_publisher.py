#!/usr/bin/env python3

import rospy

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseStamped
from copy import deepcopy

global transform

def set_transform(pose):
    global transform
    new_transform = TransformStamped()
    new_transform.header.frame_id = pose.header.frame_id
    new_transform.child_frame_id = "handover_frame"
    new_transform.transform.translation = pose.pose.position
    new_transform.transform.rotation = pose.pose.orientation
    transform = new_transform

def broadcast_transform():
    global transform
    rate = rospy.Rate(5)
    broadcaster = TransformBroadcaster()
    while not rospy.is_shutdown():
        current_transform = deepcopy(transform)
        current_transform.header.stamep = rospy.Time.now()
        broadcaster.sendTransform(current_transform)
        rate.sleep()

def main():
    global transform
    rospy.init_node("handover_frame_publisher")
    transform = TransformStamped()
    rospy.Subscriber("handover_frame_pose", PoseStamped, set_transform)
    broadcast_transform()

if __name__ == "__main__":
    main()
