#!/usr/bin/env python3

import rospy

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseStamped
from copy import deepcopy

global transform

def set_transform(pose):
    '''
    Update the current transform for the handover frame to the requested pose.
    '''
    global transform
    new_transform = TransformStamped()
    new_transform.header.frame_id = pose.header.frame_id
    new_transform.child_frame_id = "handover_frame"
    new_transform.transform.translation = pose.pose.position
    new_transform.transform.rotation = pose.pose.orientation
    transform = new_transform

def broadcast_transform():
    '''
    Constantly broadcast the current transform for the handover frame.
    '''
    global transform
    rate = rospy.Rate(5)
    broadcaster = TransformBroadcaster()
    while not rospy.is_shutdown():
        current_transform = deepcopy(transform)
        current_transform.header.stamp = rospy.Time.now()
        broadcaster.sendTransform(current_transform)
        rate.sleep()

def init_transform():
    '''
    Publish default transform during node startup so transform broadcaster can start.
    '''
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = "base_footprint"
    initial_pose.pose.orientation.w = 1
    set_transform(initial_pose)

def main():
    global transform
    rospy.init_node("handover_frame_publisher")
    transform = TransformStamped()
    init_transform()
    rospy.Subscriber("handover_frame_pose", PoseStamped, set_transform)
    broadcast_transform()

if __name__ == "__main__":
    main()
