#!/usr/bin/env python3

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, PlanningSceneInterface, RobotCommander, MoveItCommanderException
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Vector3, PointStamped, TransformStamped, PoseArray, Point
from moveit_msgs.msg import DisplayRobotState
from tf.transformations import quaternion_from_euler, quaternion_from_matrix, quaternion_multiply, quaternion_matrix, euler_from_quaternion, quaternion_inverse
import math
import numpy as np
#from gpd_ros.msg import GraspConfigList, CloudIndexed, CloudSources
#from gpd_ros.srv import detect_grasps
from std_msgs.msg import Int64
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose, do_transform_point
from bimanual_handover_msgs.srv import CollisionChecking, MoveHandover
import random
import sys
from bio_ik_msgs.msg import IKRequest, PositionGoal, DirectionGoal, AvoidJointLimitsGoal, PoseGoal
from bio_ik_msgs.srv import GetIK
from bimanual_handover.bio_ik_helper_functions import prepare_bio_ik_request, filter_joint_state
from copy import deepcopy
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from std_msgs.msg import ColorRGBA
import rosbag
import rospkg
from datetime import datetime

class HandoverMover():

    def __init__(self, debug = True, analyse = False):
        rospy.init_node('handover_mover')
        roscpp_initialize('')
        rospy.on_shutdown(self.shutdown)

        # Setup commanders
        self.hand = MoveGroupCommander("right_arm", ns = "/")
        self.hand.get_current_pose() # To initiate state monitor: see moveit issue #2715
        self.fingers = MoveGroupCommander("right_fingers", ns = "/")
        self.arm = MoveGroupCommander("right_arm_pr2", ns = "/")
        self.gripper = MoveGroupCommander("left_gripper", ns = "/")
        self.left_arm = MoveGroupCommander("left_arm", ns = "/")
        self.psi = PlanningSceneInterface(ns = "/")
        self.robot = RobotCommander()
        self.hand.set_end_effector_link("rh_manipulator")

        # Setup pc subscriber
        self.pc = None
        self.pc_sub = rospy.Subscriber("pc/pc_filtered", PointCloud2, self.update_pc)

        # Setup required services
        #rospy.wait_for_service('pc/gpd_service/detect_grasps')
        #self.gpd_service = rospy.ServiceProxy('pc/gpd_service/detect_grasps', detect_grasps)
        #rospy.wait_for_service('collision_service')
        #self.collision_service = rospy.ServiceProxy('collision_service', CollisionChecking)
        rospy.wait_for_service('/bio_ik/get_bio_ik')
        self.bio_ik_srv = rospy.ServiceProxy('/bio_ik/get_bio_ik', GetIK)

        # Setup FK
        self.fk_robot = URDF.from_parameter_server(key = "robot_description_grasp")
        self.kdl_kin_hand = KDLKinematics(self.fk_robot, "base_footprint", "rh_grasp")
        self.kdl_kin_gripper = KDLKinematics(self.fk_robot, "base_footprint", "l_gripper_tool_frame")

        # Setup transform listener and publisher
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer)
        self.handover_frame_pub = rospy.Publisher("handover_frame_pose", PoseStamped, queue_size = 1)

        # Debug
        self.debug = debug
        if self.debug:
            self.debug_gripper_pose_pub = rospy.Publisher('debug/handover_mover/gripper_pose', PoseStamped, queue_size = 1)
            self.debug_hand_pose_pub = rospy.Publisher('debug/handover_mover/hand_pose', PoseStamped, queue_size = 1)
            self.debug_sampled_transforms_pub = rospy.Publisher('debug/handover_mover/sampled_transforms', PoseArray, queue_size = 1)
            self.debug_sampled_poses_pub = rospy.Publisher('debug/handover_mover/sampled_poses', PoseArray, queue_size = 1)
            self.debug_state_pub = rospy.Publisher('debug/handover_mover/robot_state', DisplayRobotState, queue_size = 1)
            self.debug_hand_markers_pub = rospy.Publisher('debug/handover_mover/hand_markers', Marker, queue_size = 1, latch = True)
            self.debug_gripper_markers_pub = rospy.Publisher('debug/handover_mover/gripper_markers', Marker, queue_size = 1, latch = True)
        self.analyse = analyse
        if self.analyse:
            self.debug_combined_markers_pub = rospy.Publisher('debug/handover_mover/combined_markers', Marker, queue_size = 1, latch = True)
            self.time = datetime.now().strftime("%d_%m_%Y_%H_%M")
            pkg_path = rospkg.RosPack().get_path('bimanual_handover')
            path = pkg_path + "/data/bags/"
            self.bag = rosbag.Bag('{}workspace_analysis_{}.bag'.format(path, self.time), 'w')

        # Start service
        rospy.Service('handover_mover_srv', MoveHandover, self.move_handover)
        rospy.spin()

    def shutdown(self):
        if self.analyse:
            self.bag.close()
        roscpp_shutdown()

    def move_handover(self, req):
        rospy.loginfo('Request received.')
        if req.mode == "fixed":
            self.move_fixed_pose_above(req.object_type)
            return True
        elif req.mode == "above":
            self.move_fixed_pose_pc_above()
            return True
        elif req.mode == "gpd":
            self.move_gpd_pose()
            return True
        elif req.mode == "sample":
            return self.move_sampled_pose_above(req.object_type)
        else:
            rospy.loginfo("Unknown mode {}".format(req.mode))
            return False

    def update_pc(self, pc):
        self.pc = pc

    def get_sample_transformations(self):
        # Set which transformations are to sample
        translation_step = 0.03
        rotation_step = math.pi * 30/180
        if self.analyse:
            linear_combinations = [[x, y, z] for x in range(-10, 7) for y in range(-9, 20) for z in range(-13, 13)]
            angular_combinations = [[x, y, z] for x in range(-2, 3) for y in range(-2, 3) for z in range(-2, 3)]
        else:
            linear_combinations = [[x, y, z] for x in range(-1, 3) for y in range(-2, 1) for z in range(-1, 2)]
            angular_combinations = [[x, y, z] for x in range(-1, 2) for y in range(-1, 2) for z in range(-2, 3)]
        transformations = []

        # Aggregate transforms to show for debugging
        if self.debug:
            poses = PoseArray()
            poses.header.frame_id = "handover_frame"
            poses.poses = []

        # Turn parameters into transforms
        for linear in linear_combinations:
            for angular in angular_combinations:
                        new_transform = TransformStamped()
                        new_transform.header.frame_id = "handover_frame"
                        new_transform.child_frame_id = "handover_frame"
                        new_transform.transform.translation = Vector3(*[x * translation_step for x in linear])
                        new_transform.transform.rotation = Quaternion(*quaternion_from_euler(*[y * rotation_step for y in angular]))
                        transformations.append(new_transform)

                        # Add transforms for debugging
                        if self.debug:
                            new_pose = Pose()
                            new_pose.orientation = new_transform.transform.rotation
                            new_pose.position.x = new_transform.transform.translation.x
                            new_pose.position.y = new_transform.transform.translation.y
                            new_pose.position.z = new_transform.transform.translation.z
                            poses.poses.append(new_pose)
        if self.debug:
            self.debug_sampled_transforms_pub.publish(poses)
        return transformations

    def add_pose_goal(self, request, pose, eef_link):
        goal = PoseGoal()
        goal.link_name = eef_link
        goal.weight = 20.0
        goal.pose = pose.pose
        request.pose_goals = [goal]
        return request

    def add_limit_goal(self, request):
        goal = AvoidJointLimitsGoal()
        goal.weight = 10.0
        goal.primary = False
        request.avoid_joint_limits_goals = [goal]
        return request

    def add_can_goals(self, request, pose, eef_link):
        # Set the position goal
        pos_goal = PositionGoal()
        pos_goal.link_name = eef_link
        pos_goal.weight = 10.0
        pos_goal.position.x = pose.pose.position.x
        pos_goal.position.y = pose.pose.position.y
        pos_goal.position.z = pose.pose.position.z

        # Set the direction goal
        R = quaternion_matrix([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])
        z_direction = R[:3, 2]
        dir_goal = DirectionGoal()
        dir_goal.link_name = eef_link
        dir_goal.weight = 10.0
        dir_goal.axis = Vector3(0, -1, 0)
        dir_goal.direction = Vector3(-z_direction[0], -z_direction[1], -z_direction[2])

        request.position_goals = [pos_goal]
        request.direction_goals = [dir_goal]
        return request

    def score(self, joint_state):
        # Should be the maximum distance to any joint limit. Need to check joint limits if correct
        epsilon_old = math.pi

        # Calculations of delta for cost function
        delta = []
        epsilon = []
        joints = joint_state.name
        for i in range(len(joints)):
            bounds = self.robot.get_joint(joints[i]).bounds()
            delta.append(min(abs(bounds[1] - joint_state.position[i]), abs(joint_state.position[i] - bounds[0])))
            epsilon.append(abs((bounds[1] - bounds[0])/2))

        # Implementation of cost function in bimanual functional regrasping paper with corresponding epsilons
        score = 0
        for i in range(len(delta)):
            score += 1/epsilon[i]**2 * delta[i]**2 - 2/epsilon[i] * delta[i] + 1
        score = score/len(delta)

        # Same as above with fixed epsilon
        score_old = 0
        for i in range(len(delta)):
            score_old += 1/epsilon_old**2 * delta[i]**2 - 2/epsilon_old * delta[i] + 1
        score_old = score_old/len(delta)

        min_score = 1 - min(delta)

        if self.debug:
            rospy.loginfo("Score with old epsilon: {}".format(score_old))
            rospy.loginfo("Score with new epsilon: {}".format(score))
            rospy.loginfo("Min score: {}".format(min_score))

        return score

    def check_pose(self, gripper_pose, hand_pose):
        # Get hand solution
        request = prepare_bio_ik_request("right_arm", self.robot.get_current_state(), "/handover/robot_description_grasp", timeout_seconds = 0.05)
        # request.approximate = False
        # request = self.add_can_goals(request, hand_pose, 'rh_grasp')
        request = self.add_pose_goal(request, hand_pose, 'rh_grasp')
        #request = self.add_limit_goal(request)
        result = self.bio_ik_srv(request).ik_response
        # If result is not feasible, no further checking necessary, return worst score
        if result.error_code.val != 1:
            if self.debug:
                rospy.loginfo("Hand failed.")
                display_state = DisplayRobotState()
                display_state.state.joint_state = result.solution.joint_state
                self.debug_state_pub.publish(display_state)
            score = 1
            return score, None, None
        else:
            filtered_joint_state_hand = filter_joint_state(result.solution.joint_state, self.hand)
            hand_fitness = result.solution_fitness
        result = None

        # Get gripper solution
        request = prepare_bio_ik_request("left_arm", self.robot.get_current_state(), timeout_seconds = 0.05)
        # request.approximate = False
        request = self.add_pose_goal(request, gripper_pose, 'l_gripper_tool_frame')
        #request = self.add_limit_goal(request)
        result = self.bio_ik_srv(request).ik_response
        # If result is not feasible, no further checking necessary, return worst score
        if result.error_code.val != 1:
            if self.debug:
                rospy.loginfo("Gripper failed.")
            score = 1
            return score, None, None
        else:
            filtered_joint_state_gripper = filter_joint_state(result.solution.joint_state, self.left_arm)
            gripper_fitness = result.solution_fitness

        # Combine joint_states for combined score
        combined_joint_state = JointState()
        combined_joint_state.name = filtered_joint_state_hand.name + filtered_joint_state_gripper.name
        combined_joint_state.position = filtered_joint_state_hand.position + filtered_joint_state_gripper.position

        if self.debug:
            rospy.loginfo("Score hand: {}".format(self.score(filtered_joint_state_hand)))
            rospy.loginfo("Score gripper: {}".format(self.score(filtered_joint_state_gripper)))
            rospy.loginfo("Hand fitness: {}".format(hand_fitness))
            rospy.loginfo("Gripper fitness: {}".format(gripper_fitness))

        return self.score(combined_joint_state), filtered_joint_state_hand, filtered_joint_state_gripper

    def send_handover_frame(self, gripper_pose, hand_pose):
        rospy.sleep(1)
        frame_pose = PoseStamped()
        frame_pose.header.frame_id = "base_footprint"
        frame_pose.pose.orientation.w = 1
        x = min(gripper_pose.pose.position.x, hand_pose.pose.position.x) + abs(gripper_pose.pose.position.x - hand_pose.pose.position.x)/2
        frame_pose.pose.position.x = x
        y = min(gripper_pose.pose.position.y, hand_pose.pose.position.y) + abs(gripper_pose.pose.position.y - hand_pose.pose.position.y)/2
        frame_pose.pose.position.y = y
        z = min(gripper_pose.pose.position.z, hand_pose.pose.position.z) + abs(gripper_pose.pose.position.z - hand_pose.pose.position.z)/2
        frame_pose.pose.position.z = z
        self.handover_frame_pub.publish(frame_pose)
        rospy.sleep(1)

    def get_handover_transform(self, gripper_pose, hand_pose):
        self.send_handover_frame(gripper_pose, hand_pose)

        # Lookup transforms
        base_handover_transform = self.tf_buffer.lookup_transform("handover_frame", "base_footprint", rospy.Time(0))
        handover_base_transform = self.tf_buffer.lookup_transform("base_footprint", "handover_frame", rospy.Time(0))

        # Transform gripper and hand pose to handover_frame to apply transforms
        gripper_pose = do_transform_pose(gripper_pose, base_handover_transform)
        hand_pose = do_transform_pose(hand_pose, base_handover_transform)

        # Setup for iterating through transformations
        if self.analyse:
            score_limit = 0.0
        else:
            score_limit = 0.22 # old: can->0.51 book->0.56/0.57 new: can->0.2/0.21 book->0.38
        deviation_limit = 0.01
        best_score = 1
        best_transform = None
        best_hand = None
        best_gripper = None
        transformations = self.get_sample_transformations()
        if self.debug:
            debug_counter = 0
            best_debug_index = None
            poses = PoseArray()
            poses.header.frame_id = "handover_frame"
            poses.poses = []
            initial_robot_state = self.robot.get_current_state()
            hand_marker = Marker()
            hand_marker.header.frame_id = "base_footprint"
            hand_marker.ns = "hand_markers"
            hand_marker.type = Marker.POINTS
            hand_marker.action = Marker.ADD
            hand_marker.points = []
            hand_marker.colors = []
            hand_marker.scale = Vector3(0.01, 0.01, 0.01)
            gripper_marker = Marker()
            gripper_marker.header.frame_id = "base_footprint"
            gripper_marker.ns = "gripper_markers"
            gripper_marker.type = Marker.POINTS
            gripper_marker.action = Marker.ADD
            gripper_marker.points = []
            gripper_marker.colors = []
            gripper_marker.scale = Vector3(0.01, 0.01, 0.01)
        if self.analyse:
            analyse_counter = 0
            best_analyse_index = None
            previous_linear = None
            aggregated_results = []
            combined_marker = Marker()
            combined_marker.header.frame_id = "base_footprint"
            combined_marker.ns = "combined_markers"
            combined_marker.type = Marker.POINTS
            combined_marker.action = Marker.ADD
            combined_marker.points = []
            combined_marker.colors = []
            combined_marker.scale = Vector3(0.01, 0.01, 0.01)

        rospy.loginfo("Iterating through sampled transformations.")
        for transformation in transformations:
            # Transform gripper
            transformed_gripper = deepcopy(gripper_pose)
            transformed_gripper = do_transform_pose(transformed_gripper, transformation)

            # Transform hand
            transformed_hand = deepcopy(hand_pose)
            transformed_hand = do_transform_pose(transformed_hand, transformation)

            if self.debug:
                poses.poses.append(transformed_hand.pose)

            # Transform back into base_footprint for evaluation
            transformed_gripper = do_transform_pose(transformed_gripper, handover_base_transform)
            transformed_hand = do_transform_pose(transformed_hand, handover_base_transform)

            # Evaluate
            score, hand_joint_state, gripper_joint_state = self.check_pose(transformed_gripper, transformed_hand)
            if score < 1:
                hand_pos_diff, hand_quat_diff = self.calculate_fk_diff(hand_joint_state, transformed_hand, "hand")
                gripper_pos_diff, gripper_quat_diff = self.calculate_fk_diff(gripper_joint_state, transformed_gripper, "gripper")

            if self.debug:
                self.debug_gripper_pose_pub.publish(transformed_gripper)
                if not gripper_joint_state is None:
                    display_state = DisplayRobotState()
                    display_state.state = initial_robot_state
                    positions = list(display_state.state.joint_state.position)
                    for joint in gripper_joint_state.name:
                        index = gripper_joint_state.name.index(joint)
                        new_value = gripper_joint_state.position[index]
                        display_index = display_state.state.joint_state.name.index(joint)
                        positions[display_index] = new_value
                    positions = tuple(positions)
                    display_state.state.joint_state.position = positions
                    self.debug_state_pub.publish(display_state)
                hand_marker.points.append(transformed_hand.pose.position)
                gripper_marker.points.append(transformed_gripper.pose.position)
                if score < 1:
                    hand_pos_diff, hand_quat_diff = self.calculate_fk_diff(hand_joint_state, transformed_hand, "hand")
                    gripper_pos_diff, gripper_quat_diff = self.calculate_fk_diff(gripper_joint_state, transformed_gripper, "gripper")
                    if hand_pos_diff > deviation_limit:
                        hand_marker.colors.append(ColorRGBA(1, 1, 0, 1))
                    else:
                        hand_marker.colors.append(ColorRGBA(0, 1, 0, 1))
                    if gripper_pos_diff > deviation_limit:
                        gripper_marker.colors.append(ColorRGBA(1, 1, 0, 1))
                    else:
                        gripper_marker.colors.append(ColorRGBA(0, 1, 0, 1))
                else:
                    hand_marker.colors.append(ColorRGBA(1, 0, 0, 1))
                    gripper_marker.colors.append(ColorRGBA(1, 0, 0, 1))

            if score < best_score and not(hand_pos_diff >= deviation_limit or gripper_pos_diff >= deviation_limit):
                best_score = score
                best_transform = transformation
                best_hand = hand_joint_state
                best_gripper = gripper_joint_state
                if self.analyse:
                    best_analyse_index = deepcopy(analyse_counter)
                if self.debug:
                    best_debug_index = deepcopy(debug_counter)

            if self.analyse:
                if previous_linear is None:
                    previous_linear = transformation.transform.translation
                    x = min(transformed_gripper.pose.position.x, transformed_hand.pose.position.x) + abs(transformed_gripper.pose.position.x - transformed_hand.pose.position.x)/2
                    y = min(transformed_gripper.pose.position.y, transformed_hand.pose.position.y) + abs(transformed_gripper.pose.position.y - transformed_hand.pose.position.y)/2
                    z = min(transformed_gripper.pose.position.z, transformed_hand.pose.position.z) + abs(transformed_gripper.pose.position.z - transformed_hand.pose.position.z)/2
                    combined_pose = PoseStamped()
                    combined_pose.header.frame_id = "base_footprint"
                    combined_pose.pose.position = Point(x, y, z)
                    combined_pose.pose.orientation.w = 1.0
                    combined_marker.points.append(combined_pose.pose.position)
                    if score == 1:
                        aggregated_results.append(2)
                    elif self.debug and (hand_pos_diff >= deviation_limit or gripper_pos_diff >= deviation_limit):
                        aggregated_results.append(1)
                    else:
                        aggregated_results.append(0)
                elif not previous_linear == transformation.transform.translation:
                    previous_linear = transformation.transform.translation
                    x = min(transformed_gripper.pose.position.x, transformed_hand.pose.position.x) + abs(transformed_gripper.pose.position.x - transformed_hand.pose.position.x)/2
                    y = min(transformed_gripper.pose.position.y, transformed_hand.pose.position.y) + abs(transformed_gripper.pose.position.y - transformed_hand.pose.position.y)/2
                    z = min(transformed_gripper.pose.position.z, transformed_hand.pose.position.z) + abs(transformed_gripper.pose.position.z - transformed_hand.pose.position.z)/2
                    combined_pose = PoseStamped()
                    combined_pose.header.frame_id = "base_footprint"
                    combined_pose.pose.position = Point(x, y, z)
                    combined_pose.pose.orientation.w = 1.0
                    combined_marker.points.append(combined_pose.pose.position)

                    if score == 1:
                        aggregated_results.append(2)
                    elif self.debug and (hand_pos_diff >= deviation_limit or gripper_pos_diff >= deviation_limit):
                        aggregated_results.append(1)
                    else:
                        aggregated_results.append(0)

                    if all(value >= 1  for value in aggregated_results):
                        combined_marker.colors.append(ColorRGBA(1, 0, 0, 1))
                    elif sum([value == 0 for value in aggregated_results]) > len(aggregated_results)/2:
                        combined_marker.colors.append(ColorRGBA(0, 1, 0, 1))
                    elif 0 in aggregated_results:
                        combined_marker.colors.append(ColorRGBA(1, 1, 0, 1))

                    aggregated_results = []
                    analyse_counter += 1
                else:
                    if score == 1:
                        aggregated_results.append(2)
                    elif self.debug and (hand_pos_diff >= deviation_limit or gripper_pos_diff >= deviation_limit):
                        aggregated_results.append(1)
                    else:
                        aggregated_results.append(0)

            if self.debug:
                rospy.loginfo("Transform score: {}".format(score))
                debug_counter += 1

            # Stop if score already good enough
            if best_score < score_limit:
                break

        if self.debug:
            hand_marker.colors[best_debug_index] = ColorRGBA(0, 0, 1, 1)
            gripper_marker.colors[best_debug_index] = ColorRGBA(0, 0, 1, 1)
            rospy.loginfo("Best score: {}".format(best_score))
            rospy.loginfo("Best transform: {}".format(best_transform))
            self.debug_sampled_poses_pub.publish(poses)
            self.debug_hand_markers_pub.publish(hand_marker)
            self.debug_gripper_markers_pub.publish(gripper_marker)

        if self.analyse:
            if all(value >= 1  for value in aggregated_results):
                combined_marker.colors.append(ColorRGBA(1, 0, 0, 1))
            elif sum([value == 0 for value in aggregated_results]) > len(aggregated_results)/2:
                combined_marker.colors.append(ColorRGBA(0, 1, 0, 1))
            elif 0 in aggregated_results:
                combined_marker.colors.append(ColorRGBA(1, 1, 0, 1))
            combined_marker.colors[best_analyse_index] = ColorRGBA(0, 0, 1, 1)
            self.bag.write('combined', combined_marker)
            self.bag.write('gripper', gripper_marker)
            self.bag.write('hand', hand_marker)
            self.debug_combined_markers_pub.publish(combined_marker)
        return best_transform, best_hand, best_gripper

    def calculate_fk_diff(self, joint_values, target_pose, chain_type):
        target = list(joint_values.position)
        target = [self.robot.get_joint("torso_lift_joint").value()] + target
        if chain_type == "hand":
            pose_fk = self.kdl_kin_hand.forward(target, base_link = "base_footprint", end_link = "rh_grasp")
        elif chain_type == "gripper":
            pose_fk = self.kdl_kin_gripper.forward(target, base_link = "base_footprint", end_link = "l_gripper_tool_frame")

        # Calculate pos dist
        fk_pos_list = pose_fk[:3, 3].flatten().tolist()[0]
        target_pos_list = [target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z]
        pos_dist = math.dist(target_pos_list, fk_pos_list)

        # Calculate quat dist
        fk_quat = quaternion_from_matrix(pose_fk)
        target_quat = [target_pose.pose.orientation.x, target_pose.pose.orientation.y, target_pose.pose.orientation.z, target_pose.pose.orientation.w]
        inverse_fk_quat = quaternion_inverse(fk_quat)
        quat_dist = quaternion_multiply(target_quat, inverse_fk_quat)

        return pos_dist, quat_dist

    def setup_fingers(self):
        # Values except for thumb taken from spread hand.
        joint_values = dict(rh_THJ4=1.13446, rh_LFJ4=-0.31699402670752413, rh_FFJ4=-0.23131151280059523, rh_MFJ4=0.008929532157657268, rh_RFJ4=-0.11378487918959583)
        self.fingers.set_joint_value_target(joint_values)
        self.fingers.go()

    def get_gripper_pose(self):
        gripper_pose = PoseStamped()
        gripper_pose.header.frame_id = "base_footprint"
        gripper_pose.pose.position.x = 0.4753863391864514
        gripper_pose.pose.position.y = 0.03476345653124885
        gripper_pose.pose.position.z = 0.6746350873056409
        gripper_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -1.5708))
        return gripper_pose

    def move_sampled_pose_above(self, object_type = None):
        self.setup_fingers()

        # Pose through initial intuition, just from where to start sampling
        gripper_pose = self.get_gripper_pose()

        # Setup hand pose
        hand_pose = deepcopy(gripper_pose)
        if object_type == "book":
            hand_pose.pose.position.y += -0.02
            hand_pose.pose.position.x += -0.03
            hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(-1.5708, 3.14159, -1.5706))
        else:
            hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(-1.5708, 3.14159, 0))
        hand_pose.pose.position.z += 0.167
        pre_hand_pose = deepcopy(hand_pose)
        pre_hand_pose.pose.position.z += 0.08

        if self.debug:
            self.debug_hand_pose_pub.publish(hand_pose)

        # Get required transforms
        handover_transform, hand_joint_state, gripper_joint_state = self.get_handover_transform(gripper_pose, hand_pose)
        base_handover_transform = self.tf_buffer.lookup_transform("handover_frame", "base_footprint", rospy.Time(0))
        handover_base_transform = self.tf_buffer.lookup_transform("base_footprint", "handover_frame", rospy.Time(0))

        if not handover_transform is None:
            rospy.loginfo("Handover pose found.")
        else:
            rospy.logerr("No handover pose found.")
            return False

        # Transform poses into handover_frame
        gripper_pose = do_transform_pose(gripper_pose, base_handover_transform)
        hand_pose = do_transform_pose(hand_pose, base_handover_transform)
        pre_hand_pose = do_transform_pose(pre_hand_pose, base_handover_transform)

        # Apply handover_transform
        gripper_pose = do_transform_pose(gripper_pose, handover_transform)
        hand_pose = do_transform_pose(hand_pose, handover_transform)
        pre_hand_pose = do_transform_pose(pre_hand_pose, handover_transform)

        # Tranform poses back into base_footprint
        gripper_pose = do_transform_pose(gripper_pose, handover_base_transform)
        hand_pose = do_transform_pose(hand_pose, handover_base_transform)
        pre_hand_pose = do_transform_pose(pre_hand_pose, handover_base_transform)

        if self.debug:
            self.debug_gripper_pose_pub.publish(gripper_pose)

        self.left_arm.set_joint_value_target(gripper_joint_state)
        self.left_arm.go()

        '''
        # Get gripper solution
        request = prepare_bio_ik_request("left_arm", self.robot.get_current_state())
        # request.approximate = False
        request = self.add_pose_goal(request, gripper_pose, 'l_gripper_tool_frame')
        result = self.bio_ik_srv(request).ik_response
        if result.error_code.val != 1:
            rospy.logerr("Bio_ik planning request returned error code {}.".format(result.error_code.val))
            return False
        filtered_joint_state_gripper = filter_joint_state(result.solution.joint_state, self.left_arm)

        # Move gripper to requested pose
        rospy.loginfo("Moving left arm to handover pose.")
        self.left_arm.set_joint_value_target(filtered_joint_state_gripper)
        self.left_arm.go()
        '''

        if self.debug:
            self.debug_hand_pose_pub.publish(hand_pose)

        # Get hand approach solution
        request = prepare_bio_ik_request("right_arm", self.robot.get_current_state(), "/handover/robot_description_grasp")
        request = self.add_pose_goal(request, pre_hand_pose, 'rh_grasp')
        result = self.bio_ik_srv(request).ik_response
        if result.error_code.val != 1:
            rospy.logerr("Bio_ik planning request returned error code {}.".format(result.error_code.val))
            return False
        filtered_joint_state_pre_hand = filter_joint_state(result.solution.joint_state, self.hand)

        # Move hand to requested pose
        # Use bio_ik to be able to use rh_grasp as end effector
        rospy.loginfo("Moving right arm to handover pose.")
        self.hand.set_joint_value_target(filtered_joint_state_pre_hand)
        self.hand.go()

        '''
        # Get hand solution
        request = prepare_bio_ik_request("right_arm", self.robot.get_current_state(), "/handover/robot_description_grasp")
        # request.approximate = False
        request = self.add_pose_goal(request, hand_pose, 'rh_grasp')
        result = self.bio_ik_srv(request).ik_response
        if result.error_code.val != 1:
            rospy.logerr("Bio_ik planning request returned error code {}.".format(result.error_code.val))
            return False
        filtered_joint_state_hand = filter_joint_state(result.solution.joint_state, self.hand)

        # Move hand to requested pose
        # Use bio_ik to be able to use rh_grasp as end effector
        rospy.loginfo("Moving right arm to handover pose.")
        self.hand.set_joint_value_target(filtered_joint_state_hand)
        self.hand.go()
        '''

        self.hand.set_joint_value_target(hand_joint_state)
        self.hand.go()

        return True

    def move_gpd_pose(self):
        while self.pc is None:
            rospy.sleep(1)
        self.setup_fingers()
        gripper_pose = self.get_gripper_pose()
        self.left_arm.set_pose_target(gripper_pose)
        self.left_arm.go()

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
            self.debug_hand_pose_pub.publish(transformed_pose)
            if self.collision_service(transformed_pose.pose, gripper_pose):
                checked_poses.append(transformed_pose)
        if not checked_poses:
            rospy.loginfo("No valid pose was found.")
            return False
        for pose in checked_poses:
            try:
                self.hand.set_pose_target(pose)
                result = self.hand.go()
                if result:
                    rospy.loginfo("hand moved")
                    break
            except MoveItCommanderException as e:
                print(e)

    def move_fixed_pose_above(self, object_type = None):
        self.setup_fingers()
        gripper_pose = self.get_gripper_pose()
        self.left_arm.set_pose_target(gripper_pose)
        self.left_arm.go()
        rospy.sleep(1)

        # Calculate desired position and direction relative to l_gripper_tool_frame in base_footprint
        gripper_pose = self.gripper.get_current_pose(end_effector_link = "l_gripper_tool_frame")
        R = quaternion_matrix([gripper_pose.pose.orientation.x, gripper_pose.pose.orientation.y, gripper_pose.pose.orientation.z, gripper_pose.pose.orientation.w])
        y_direction = R[:3, 1]
        z_direction = R[:3, 2]
        gripper_base_transform = self.tf_buffer.lookup_transform("base_footprint", "l_gripper_tool_frame", rospy.Time(0))
        transformed_pos = PointStamped()
        transformed_pos.header.frame_id = "l_gripper_tool_frame"
        if object_type == "book":
            transformed_pos.point.x = 0.02
        else:
            transformed_pos.point.x = -0.002
        transformed_pos.point.y = 0
        transformed_pos.point.z = 0.167
        transformed_pos = do_transform_point(transformed_pos, gripper_base_transform)

        # Debug stuff
        if self.debug:
            print(transformed_pos)
            debug_pose = PoseStamped()
            debug_pose.header.frame_id = "base_footprint"
            debug_pose.pose.position.x = transformed_pos.point.x
            debug_pose.pose.position.y = transformed_pos.point.y
            debug_pose.pose.position.z = transformed_pos.point.z
            self.debug_hand_pose_pub.publish(debug_pose)

        # Prepare bio_ik request
        request = IKRequest()
        # Set non-goal parameters for the request
        # Load robot_model which has rh_grasp as an additional frame
        request.robot_description = "/handover/robot_description_grasp"
        request.group_name = "right_arm"
        request.approximate = True
        request.timeout = rospy.Duration.from_sec(1)
        request.avoid_collisions = True
        request.robot_state = self.robot.get_current_state()

        # Set the position goal
        pos_goal = PositionGoal()
        pos_goal.link_name = "rh_grasp"
        pos_goal.weight = 10.0
        pos_goal.position.x = transformed_pos.point.x
        pos_goal.position.y = transformed_pos.point.y
        pos_goal.position.z = transformed_pos.point.z

        # Set the direction goal
        dir_goal = DirectionGoal()
        dir_goal.link_name = "rh_grasp"
        dir_goal.weight = 10.0
        dir_goal.axis = Vector3(0, -1, 0)
        dir_goal.direction = Vector3(-z_direction[0], -z_direction[1], -z_direction[2])

        # Set secondary goals
        limit_goal = AvoidJointLimitsGoal()
        weight = 5.0
        primary = False

        # Add the previous goals
        request.position_goals = [pos_goal]
        request.direction_goals = [dir_goal]
        request.avoid_joint_limits_goals = [limit_goal]

        # Set additional goals for different objects
        if object_type == "book":
            dir_goal = DirectionGoal()
            dir_goal.link_name = "rh_grasp"
            dir_goal.weight = 8.0
            dir_goal.axis = Vector3(0, 0, 1)
            dir_goal.direction = Vector3(y_direction[0], y_direction[1], y_direction[2])
            request.direction_goals.append(dir_goal)

        # Get bio_ik solution
        response = self.bio_ik_srv(request).ik_response
        if not response.error_code.val == 1:
            print(response)
            raise Exception("Bio_ik planning failed with error code {}.".format(response.error_code.val))

        # Filter solution for relevante joints
        joint_names = self.hand.get_active_joints()
        joint_target_state = JointState()
        joint_target_state.name = joint_names
        joint_target_state.position = [response.solution.joint_state.position[response.solution.joint_state.name.index(joint_name)] for joint_name in joint_names]

        # Execute solution
        self.hand.set_joint_value_target(joint_target_state)
        plan = self.hand.go()
        if not plan:
            raise Exception("No path was found to the joint state \n {}.".format(joint_target_state))

    def move_fixed_pose_pc_above(self):
        self.setup_fingers()
        gripper_pose = self.get_gripper_pose()
        self.left_arm.set_pose_target(gripper_pose)
        self.left_arm.go()

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
        hand_pose.pose.position.x = min_point[0] + math.dist([max_point[0]], [min_point[0]])/2 - 0.01
        hand_pose.pose.position.y = min_point[1] + math.dist([max_point[1]], [min_point[1]])/2 - 0.07
        hand_pose.pose.position.z = max_point[2] + 0.05
        hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(1.57, 0, 3).tolist())

        # publish the hand pose if debugging is desired
        if self.debug:
            hand_pub.publish(hand_pose)

        # move the hand to the previously set pose
        self.hand.set_pose_target(hand_pose)
        self.hand.go()

        # spawn a collision object in the rough shape of the pointloud
        # Unused, maybe later for visualization or collision checking
        '''
        can_pose = hand_pose
        can_pose.pose.position.y += 0.05
        can_pose.pose.position.z = min_point[2] + math.dist([max_point[2]], [min_point[2]])/2
        can_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, 0).tolist())
        self.psi.add_cylinder('can', can_pose, math.dist([max_point[2]],[min_point[2]]), math.dist([max_point[0]], [min_point[0]])/2)
        self.psi.disable_collision_detections('can', self.robot.get_link_names("right_fingers"))
        self.psi.disable_collision_detections('can', ['rh_ff_biotac_link', 'rh_mf_biotac_link', 'rh_rf_biotac_link', 'rh_lf_biotac_link', 'rh_th_biotac_link', 'rh_ffdistal', 'rh_mfdistal', 'rh_rfdistal', 'rh_lfdistal', 'rh_thdistal'])
        '''
        return True

if __name__ == "__main__":
    mover = HandoverMover()

