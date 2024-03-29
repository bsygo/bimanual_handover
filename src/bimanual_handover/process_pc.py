#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.srv import ProcessPC
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import PoseStamped

global pub, received, left_arm

def process_pc(req):
    '''
    Wait for processed pc and move left arm back to initial pose once received.
    '''
    global pub, received, left_arm
    rospy.loginfo("Process_pc service request received.")
    pub.publish(req.publish)
    while not received:
        rospy.sleep(0.1)
    received = False
    target_pose = PoseStamped()
    target_pose.header.frame_id = "base_footprint"
    target_pose.pose.position.x = 0.5027
    target_pose.pose.position.y = 0.6274
    target_pose.pose.position.z = 0.6848
    target_pose.pose.orientation.w = 1.0
    left_arm.set_pose_target(target_pose)
    left_arm.go()
    return True

def pc_received(pc):
    '''
    Set mode to wait for next point cloud.
    '''
    global received
    received = True

def main():
    global pub, received, left_arm
    rospy.init_node('process_pc_stub')
    received = False
    left_arm = MoveGroupCommander("left_arm", ns = "/")
    left_arm.set_max_velocity_scaling_factor(1.0)
    left_arm.set_max_acceleration_scaling_factor(1.0)
    pub = rospy.Publisher('publish_pc', Bool, queue_size = 5)
    sub = rospy.Subscriber('pc/pc_final', PointCloud2, pc_received)
    rospy.Service('process_pc_srv', ProcessPC, process_pc)
    rospy.spin()

if __name__ == "__main__":
    main()
