#!/usr/bin/env python3

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, PlanningSceneInterface, RobotCommander
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler
import math

class RobotSetupMover():

    def __init__(self, debug = False):
        self.hand = MoveGroupCommander("right_arm")
        self.fingers = MoveGroupCommander("right_fingers")
        self.arm = MoveGroupCommander("right_arm_pr2")
        self.psi = PlanningSceneInterface()
        self.robot = RobotCommander()
        self.hand.set_end_effector_link("rh_manipulator")
        self.debug = debug
        self.pc = None
        self.pc_sub = rospy.Subscriber("/pc/pc_filtered", PointCloud2, self.update_pc)
        print(self.pc_sub)

    def update_pc(self, pc):
        self.pc = pc
   
    def move_fixed_pose_pc(self):
        finished = False
        while not finished:
            finished = self.try_move_fixed_pose_pc()

    def try_move_fixed_pose_pc(self):
        if self.pc is None:
            return False
        pub = rospy.Publisher("debug_marker", Marker, latch = True, queue_size = 1)
        hand_pub = rospy.Publisher("debug_hand_pose", PoseStamped, latch = True, queue_size = 1)
        gen = pc2.read_points(self.pc, field_names = ("x", "y", "z"), skip_nans = True)

        # find max and min values of the pointcloud
        max_point = [-100, -100, -100]
        min_point = [100, 100, 100]
        for point in gen:
            for x in range(len(point)):
                if point[x] > max_point[x]:
                    max_point[x] = point[x]
                if point[x] < min_point[x]:
                    min_point[x] = point[x]

        # if debugging is wanted, publish the center top of the pointcloud
        if self.debug:
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

        # set the pose for the hand to a fixed pose related to the pointcloud
        hand_pose = PoseStamped()
        hand_pose.header.frame_id = "base_footprint"
        hand_pose.pose.position.x = min_point[0] + math.dist([max_point[0]], [min_point[0]])/2
        hand_pose.pose.position.y = min_point[1] + math.dist([max_point[1]], [min_point[1]])/2 - 0.05
        hand_pose.pose.position.z = max_point[2] + 0.03
        hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(1.57, 0, 3).tolist())

        # publish the hand pose if debugging is desired
        if self.debug:
            hand_pub.publish(hand_pose)

        # move the hand to the previously set pose
        self.hand.set_pose_target(hand_pose)
        self.hand.go()
        self.hand.stop()

        # spawn a collision object in the rough shape of the pointloud
        can_pose = hand_pose
        can_pose.pose.position.y += 0.05
        can_pose.pose.position.z = min_point[2] + math.dist([max_point[2]], [min_point[2]])/2
        can_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, 0).tolist())
        self.psi.add_cylinder('can', can_pose, math.dist([max_point[2]],[min_point[2]]), math.dist([max_point[0]], [min_point[0]])/2)
        self.psi.disable_collision_detections('can', self.robot.get_link_names("right_fingers"))
        self.psi.disable_collision_detections('can', ['rh_ff_biotac_link', 'rh_mf_biotac_link', 'rh_rf_biotac_link', 'rh_lf_biotac_link', 'rh_th_biotac_link', 'rh_ffdistal', 'rh_mfdistal', 'rh_rfdistal', 'rh_lfdistal', 'rh_thdistal'])
        return True

    def reset_fingers(self):
        self.fingers.set_named_target('open')
        self.fingers.go()

    def reset_arm(self):
        self.arm.set_named_target('right_arm_to_side')
        self.arm.go()

if __name__ == "__main__":
    rospy.init_node('robot_setup_mover')
    mover = RobotSetupMover()
    mover.reset_fingers()
    mover.move_fixed_pose_pc()
