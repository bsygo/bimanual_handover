#!/usr/bin/env python3

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler
import math

global hand

def move(pc):
    pub = rospy.Publisher("debug_marker", Marker, latch = True, queue_size = 1)
    hand_pub = rospy.Publisher("debug_hand_pose", PoseStamped, latch = True, queue_size = 1)
    gen = pc2.read_points(pc, field_names = ("x", "y", "z"), skip_nans = True)
    max_point = [-100, -100, -100]
    min_point = [100, 100, 100]
    for point in gen:
        for x in range(len(point)):
            if point[x] > max_point[x]:
                max_point[x] = point[x]
            if point[x] < min_point[x]:
                min_point[x] = point[x]
    print(min_point)
    print(max_point)
    marker = Marker()
    marker.header.frame_id = "base_footprint"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = min_point[0] + math.dist([max_point[0]], [min_point[0]])/2
    marker.pose.position.y = min_point[1] + math.dist([max_point[1]], [min_point[1]])/2
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
    hand_pose = PoseStamped()
    hand_pose.header.frame_id = "base_footprint"
    hand_pose.pose.position.x = min_point[0] + math.dist([max_point[0]], [min_point[0]])/2
    hand_pose.pose.position.y = min_point[1] + math.dist([max_point[1]], [min_point[1]])/2 - 0.025
    hand_pose.pose.position.z = max_point[2] + 0.02
    hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(1.57, 0, 3).tolist())
    hand_pub.publish(hand_pose)
    hand.set_pose_target(hand_pose)
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
    hand.set_end_effector_link("rh_manipulator")
    rospy.Subscriber("/pc/pc_filtered", PointCloud2, move)
    rospy.spin()


if __name__ == "__main__":
    main()
