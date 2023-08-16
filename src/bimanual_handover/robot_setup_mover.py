#!/usr/bin/env python3

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, PlanningSceneInterface, RobotCommander, MoveItCommanderException
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Vector3
from tf.transformations import quaternion_from_euler, quaternion_from_matrix, quaternion_multiply
import math
#from gpd_ros.msg import GraspConfigList, CloudIndexed, CloudSources
#from gpd_ros.srv import detect_grasps
from bimanual_handover.srv import CollisionChecking
from std_msgs.msg import Int64
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
from bimanual_handover.srv import CollisionChecking, MoveHandover
import random
import sys

class RobotSetupMover():

    def __init__(self, debug = False):
        rospy.init_node('robot_setup_mover')
        roscpp_initialize('')
        rospy.on_shutdown(self.shutdown)
        self.hand = MoveGroupCommander("right_arm", ns = "/")
        self.fingers = MoveGroupCommander("right_fingers", ns = "/")
        self.arm = MoveGroupCommander("right_arm_pr2", ns = "/")
        self.gripper = MoveGroupCommander("left_gripper", ns = "/")
        self.psi = PlanningSceneInterface(ns = "/")
        self.robot = RobotCommander()
        self.hand.set_end_effector_link("rh_manipulator")
        self.debug = debug
        self.pc = None
        self.pc_sub = rospy.Subscriber("pc/pc_filtered", PointCloud2, self.update_pc)
        #rospy.wait_for_service('pc/gpd_service/detect_grasps')
        #self.gpd_service = rospy.ServiceProxy('pc/gpd_service/detect_grasps', detect_grasps)
        self.debug_pose_pub = rospy.Publisher('debug/rsm_setup_pose', PoseStamped, queue_size = 1)
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer)
        #rospy.wait_for_service('/cc/collision_service')
        #self.collision_service = rospy.ServiceProxy('/cc/collision_service', CollisionChecking)
        #self.sh_pose_pub = rospy.Publisher('/cc/sh_pose', Pose)
        #self.gripper_pose_pub = rospy.Publisher('/cc/gripper_pose', Pose)
        rospy.Service('move_handover_srv', MoveHandover, self.move_handover)
        rospy.spin()

    def shutdown(self):
        roscpp_shutdown()

    def move_handover(self, req):
        rospy.loginfo('Request received.')
        if req.mode == "fixed":
            self.move_fixed_pose_pc()
            return True
        elif req.mode == "gpd":
            self.move_gpd_pose()
            return True
        else:
            rospy.loginfo("Unknown mode {}".format(req.mode))
            return False

    def update_pc(self, pc):
        self.pc = pc

    def sample_transformation(self):
        translation = Vector3()
        translation.x = random.randrange(-0.5, 0.5)
        translation.y = random.randrange(0, 1)
        translation.z = random.randrange(-0.5, 0.5)
        #rotation = *quaternion_from_euler((random.randrange(-math.pi/2, math.pi/2), random.randrange(-math.pi/2, math.pi/2), random.randrange(-math.pi/2, math.pi/2)))
        return translation, rotation

    def move_gpd_pose(self):
        while self.pc is None:
            rospy.sleep(1)
        self.fingers.set_joint_value_target('rh_THJ4, 1.13446')
        self.fingers.go()
        current_pc = self.pc
        cloud_indexed = CloudIndexed()
        cloud_sources = CloudSources()
        cloud_sources.cloud = current_pc
        cloud_sources.view_points = [self.robot.get_link('azure_kinect_rgb_camera_link_urdf').pose().pose.position]
        cloud_sources.camera_source = [Int64(0) for i in range(current_pc.width)]
        cloud_indexed.cloud_sources = cloud_sources
        cloud_indexed.indices = [Int64(i) for i in range(current_pc.width)]
        response = self.gpd_service(cloud_indexed)
        manipulator_transform = self.tf_buffer.lookup_transform('rh_manipulator', 'base_footprint', rospy.Time(0))
        checked_poses = []
        gripper_pose = self.gripper.get_current_pose()
        '''
        # Use the middle point between thumb and middle finger as the grasp point for the hand
        mf_pose = self.fingers.get_current_pose('rh_mf_biotac_link')
        th_pose = self.fingers.get_current_pose('rh_th_biotac_link')
        hand_grasp_point = PoseStamped()
        hand_grasp_point.pose.position.x = (mf_pose.pose.position.x + th_pose.pose.position.x)/2
        hand_grasp_point.pose.position.y = (mf_pose.pose.position.y + th_pose.pose.position.y)/2
        hand_grasp_point.pose.position.z = (mf_pose.pose.position.z + th_pose.pose.position.z)/2
        grasp_point_transform = self.tf_buffer.lookup_transform('base_footprint', 'rh_manipulator')
        hand_grasp_point = do_transform_pose(hand_grasp_point, grasp_point_transform)
        '''
        for i in range(len(response.grasp_configs.grasps)):
            selected_grasp = response.grasp_configs.grasps[i]
            grasp_point = selected_grasp.position
            R = [[selected_grasp.approach.x, selected_grasp.binormal.x, selected_grasp.axis.x, 0], [selected_grasp.approach.y, selected_grasp.binormal.y, selected_grasp.axis.y, 0], [selected_grasp.approach.z, selected_grasp.binormal.z, selected_grasp.axis.z, 0], [0, 0, 0, 1]]
            grasp_q = quaternion_from_matrix(R)
            sh_q = quaternion_from_euler(math.pi/2, -math.pi/2, 0)
            final_q = Quaternion(*quaternion_multiply(grasp_q, sh_q))
            pose = PoseStamped()
            pose.header.frame_id = "base_footprint"
            pose.pose.position = grasp_point
            pose.pose.orientation = final_q
            transformed_pose = do_transform_pose(pose, manipulator_transform)
            transformed_pose.pose.position.y += 0.02
            transformed_pose.pose.position.z += - 0.02
            '''
            # Adjust hand movement from rh_manipulator to the grasp point specified above
            transformed_pose.pose.position.x += hand_grasp_point.pose.position.x
            transformed_pose.pose.position.y += hand_grasp_point.pose.position.y
            transformed_pose.pose.position.z += hand_grasp_point.pose.position.z
            '''
            self.debug_pose_pub.publish(transformed_pose)
            if self.collision_service(transformed_pose.pose, gripper_pose):
                checked_poses.append(transformed_pose)
        if not checked_poses:
            rospy.loginfo("No valid pose was found.")
            return
        for pose in checked_poses:
            try:
                self.hand.set_pose_target(pose)
                result = self.hand.go()
                if result:
                    rospy.loginfo("hand moved")
                    break
            except MoveItCommanderException as e:
                print(e)

    def setup_fingers(self):
        # Values except for thumb taken from spread hand.
        joint_values = dict(rh_THJ4=1.13446, rh_LFJ4=-0.31699402670752413, rh_FFJ4=-0.23131151280059523, rh_MFJ4=0.008929532157657268, rh_RFJ4=-0.11378487918959583) 
        self.fingers.set_joint_value_target(joint_values)
        self.fingers.go()

    def move_fixed_pose_pc(self):
        self.setup_fingers()
        rospy.loginfo('Moving to fixed pose.')
        while self.pc is None:
            rospy.sleep(1)
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

if __name__ == "__main__":
    mover = RobotSetupMover()

