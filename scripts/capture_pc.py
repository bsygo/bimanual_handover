#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
import sys
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose

global pub, publish, left_arm, tf_buffer, debug_pub, pub_time

def save_pc(new_pc):
    global pub, publish, pub_time
    # Only publish when requested and pc is recent
    if publish and (new_pc.header.stamp > pub_time):
        pub.publish(new_pc)
        publish = False

def publish_pc(publish_pc):
    global publish, left_arm, tf_buffer, debug_pub, pub_time
    rospy.loginfo("Publish_pc request received.")

    camera_base_transform = tf_buffer.lookup_transform("base_footprint", "azure_kinect_rgb_camera_link", rospy.Time(0))
    pc_pose = PoseStamped()
    pc_pose.header.frame_id = "azure_kinect_rgb_camera_link"
    pc_pose.pose.position.x = 0
    pc_pose.pose.position.y = 0
    pc_pose.pose.position.z = 0.6
    pc_pose.pose.orientation = Quaternion(*quaternion_from_euler(2.453, 0, 0))
    pc_pose = do_transform_pose(pc_pose, camera_base_transform)

    debug_pub.publish(pc_pose)

    left_arm.set_pose_target(pc_pose)
    left_arm.go()

    # To wait out the camera delay
    pub_time = rospy.Time.now()

    publish = publish_pc

def main():
    global pub, publish, left_arm, tf_buffer, debug_pub
    rospy.init_node('capture_pc')

    tf_buffer = Buffer()
    TransformListener(tf_buffer)
    left_arm = MoveGroupCommander("left_arm", ns = "/")
    left_arm.set_max_velocity_scaling_factor(1.0)
    left_arm.set_max_acceleration_scaling_factor(1.0)

    if len(sys.argv) < 2:
        rospy.logerr("Missing argument for capture_pc.")
        return
    elif sys.argv[1] == "true":
        input_topic = 'pc/cloud_pcd'
    elif sys.argv[1] == "false":
        input_topic = '/azure_kinect/points2'
    else:
        rospy.logerr("Unknown argument {} for capture_pc.".format(sys.argv[1]))
        return

    publish = False
    rospy.Subscriber(input_topic, PointCloud2, save_pc, queue_size = 1)
    rospy.Subscriber('publish_pc', Bool, publish_pc, queue_size = 1)
    pub = rospy.Publisher('pc/pc_raw', PointCloud2, queue_size=5)
    debug_pub = rospy.Publisher('debug/pc_capture/gripper_pose', PoseStamped, queue_size = 1, latch = True)
    rospy.spin()

if __name__ == "__main__":
    main()
