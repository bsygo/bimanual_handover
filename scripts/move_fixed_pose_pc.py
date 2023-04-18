#!/usr/bin/env python3

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

global hand

def move(pc):
    pub = rospy.Publisher("debug_marker", Marker, latch = True, queue_size = 1)
    gen = pc2.read_points(pc, field_names = ("x", "y", "z"), skip_nans = True)
    max_value = -100
    max_point = (0, 0, 0)
    for point in gen:
        if point[2] > max_value:
            max_value = point[2]
            max_point = point
    print(max_value)
    print(max_point)
    marker = Marker()
    marker.header.frame_id = "base_footprint"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = max_point[0]
    marker.pose.position.y = max_point[1]
    marker.pose.position.z = max_point[2]
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.z = 1.0
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1
    marker.color.r = 1.0
    marker.color.g = 0
    marker.color.b = 0
    pub.publish(marker)
    '''
    hand_pose = PoseStamped()
    hand_pose.pose.position.x = max_point[0]
    hand_pose.pose.position.y = max_point[1]
    hand_pose.pose.position.z = max_point[2]
    hand_pose.pose.orientation = 
    hand.set_pose_target()
    '''
    hand.set_position_target(list(max_point))
    hand.go()
    hand.stop()

def end_moveit():
    roscpp_shutdown()

def main():
    global hand
    rospy.init_node("pc_mover")
    rospy.on_shutdown(end_moveit)
    roscpp_initialize("")
    hand = MoveGroupCommander("right_arm")
    rospy.Subscriber("/pc/pc_filtered", PointCloud2, move)
    rospy.spin()


if __name__ == "__main__":
    main()
