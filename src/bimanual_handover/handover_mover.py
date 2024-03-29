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
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension

class HandoverMover():

    def __init__(self):
        rospy.init_node('handover_mover')
        roscpp_initialize('')
        rospy.on_shutdown(self.shutdown)

        # Setup commanders
        self.hand = MoveGroupCommander("right_arm", ns = "/")
        self.hand.get_current_pose() # To initiate state monitor: see moveit issue #2715
        self.fingers = MoveGroupCommander("right_fingers", ns = "/")
        self.fingers.set_max_velocity_scaling_factor(1.0)
        self.fingers.set_max_acceleration_scaling_factor(1.0)
        self.arm = MoveGroupCommander("right_arm_pr2", ns = "/")
        self.gripper = MoveGroupCommander("left_gripper", ns = "/")
        self.left_arm = MoveGroupCommander("left_arm", ns = "/")
        self.left_arm.set_max_velocity_scaling_factor(1.0)
        self.left_arm.set_max_acceleration_scaling_factor(1.0)
        self.psi = PlanningSceneInterface(ns = "/")
        self.robot = RobotCommander()
        self.hand.set_end_effector_link("rh_manipulator")

        # Setup pc subscriber
        self.pc = None
        self.pc_sub = rospy.Subscriber("pc/pc_final", PointCloud2, self.update_pc)

        # Setup required services
        rospy.wait_for_service('/bio_ik/get_bio_ik')
        self.bio_ik_srv = rospy.ServiceProxy('/bio_ik/get_bio_ik', GetIK)

        # Initialize settings for current handover
        self.side = None
        self.grasp_pose_mode = None
        self.handover_pose_mode = None
        self.object_type = None

        # Setup FK
        self.fk_robot = URDF.from_parameter_server(key = "/robot_description")
        self.kdl_kin_hand = KDLKinematics(self.fk_robot, "base_footprint", "rh_manipulator")
        self.kdl_kin_gripper = KDLKinematics(self.fk_robot, "base_footprint", "l_gripper_tool_frame")

        # Setup transform listener and publisher
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer)
        self.handover_frame_pub = rospy.Publisher("handover_frame_pose", PoseStamped, queue_size = 1)

        # Debug
        self.pkg_path = rospkg.RosPack().get_path('bimanual_handover')
        self.verbose = rospy.get_param("handover_mover/verbose")
        self.debug = rospy.get_param("handover_mover/debug")
        self.write_time = rospy.get_param("handover_mover/write_time")
        if self.write_time:
            self.time = datetime.now().strftime("%d_%m_%Y_%H_%M")
            time_path = self.pkg_path + "/data/records/"
            self.time_file = open('{}sampling_times_{}.txt'.format(time_path, self.time), 'w')
        if self.debug:
            self.debug_gripper_pose_pub = rospy.Publisher('debug/handover_mover/gripper_pose', PoseStamped, queue_size = 1, latch = True)
            self.debug_hand_pose_pub = rospy.Publisher('debug/handover_mover/hand_pose', PoseStamped, queue_size = 1, latch = True)
            self.debug_sampled_transforms_pub = rospy.Publisher('debug/handover_mover/sampled_transforms', PoseArray, queue_size = 1, latch = True)
            self.debug_sampled_poses_pub = rospy.Publisher('debug/handover_mover/sampled_poses', PoseArray, queue_size = 1, latch = True)
            self.debug_state_pub = rospy.Publisher('debug/handover_mover/robot_state', DisplayRobotState, queue_size = 1, latch = True)
            self.debug_gripper_state_pub = rospy.Publisher('debug/handover_mover/gripper_state', DisplayRobotState, queue_size = 1, latch = True)
            self.debug_hand_state_pub = rospy.Publisher('debug/handover_mover/hand_state', DisplayRobotState, queue_size = 1, latch = True)
            self.debug_hand_markers_pub = rospy.Publisher('debug/handover_mover/hand_markers', Marker, queue_size = 1, latch = True)
            self.debug_gripper_markers_pub = rospy.Publisher('debug/handover_mover/gripper_markers', Marker, queue_size = 1, latch = True)
            self.debug_handover_marker_pub = rospy.Publisher('debug/handover_mover/handover_marker', Marker, queue_size = 1, latch = True)

        # Start service
        rospy.Service('handover_mover_srv', MoveHandover, self.move_handover)
        rospy.spin()

    def shutdown(self):
        if self.write_time:
            self.time_file.close()
        roscpp_shutdown()

    def move_handover(self, req):
        '''
        Handle received requests to move to handover pose.
        '''
        rospy.loginfo('Request received.')
        # Set parameters to requested values.
        self.side = req.side
        self.grasp_pose_mode = req.grasp_pose_mode
        self.handover_pose_mode = req.handover_pose_mode
        self.object_type = req.object_type

        # Call appropriate function for given grasp pose mode.
        if self.grasp_pose_mode == "fixed":
            hand_pose = self.get_hand_pose_fixed()
        elif self.grasp_pose_mode == "pc":
            hand_pose = self.get_hand_pose_pc()
        else:
            rospy.loginfo("Unknown grasp_pose_mode {}".format(req.grasp_pose_mode))
            return False, None

        # Call appropriate function for given handover pose mode.
        if req.handover_pose_mode == "fixed":
            return self.move_fixed_pose(hand_pose)
        elif req.handover_pose_mode == "random_sample" or req.handover_pose_mode == "load_sample":
            return self.move_sampled_pose(hand_pose)
        else:
            rospy.loginfo("Unknown mode {}".format(req.handover_pose_mode))
            return False, None

    def update_pc(self, pc):
        '''
        Update point cloud whenever a new one is received.
        '''
        self.pc = pc

    def load_transforms(self):
        '''
        Get the transforms from the specified bag file.
        '''
        filename = rospy.get_param("handover_mover/sample_file")
        transforms = []
        transforms_bag = rosbag.Bag(self.pkg_path + "/data/workspace_analysis/" + filename, 'r')
        for _, msg, _ in transforms_bag.read_messages():
            transforms.append(msg)
        transforms_bag.close()
        random.shuffle(transforms)
        return transforms
        # Commented part was used for screenshots
        '''
        filtered_transforms = []
        for transform in transforms:
            if transform.transform.translation.x == 0.12 and transform.transform.translation.y == 0.06 and transform.transform.translation.z == 0.0:
                filtered_transforms.append(transform)
        return filtered_transforms
        '''

    def get_random_sample_transformations(self, number_transforms):
        '''
        Get a list of random transforms
        '''
        translation_step = 0.06
        # Values from workspace analysis
        if self.object_type == "can" and self.side == "side":
            min_linear_limits = [0, 2, -4]
            min_linear_limits = [translation_step * limit for limit in min_linear_limits]
            max_linear_limits = [2, 5, 0]
            max_linear_limits = [translation_step * limit for limit in max_linear_limits]
        else:
            min_linear_limits = [0, 0, -3]
            min_linear_limits = [translation_step * limit for limit in min_linear_limits]
            max_linear_limits = [1, 4, 0]
            max_linear_limits = [translation_step * limit for limit in max_linear_limits]

        rotation_step = math.pi * 30/180
        min_angular_limits = [-3, -3, -3]
        min_angular_limits = [rotation_step * limit for limit in min_angular_limits]
        max_angular_limits = [3, 3, 3]
        max_angular_limits = [rotation_step * limit for limit in max_angular_limits]

        transformations = []
        for i in range(number_transforms):
            new_transform = TransformStamped()
            new_transform.header.frame_id = "handover_frame"
            new_transform.child_frame_id = "handover_frame"
            new_transform.transform.translation = Vector3(*[random.uniform(min_linear_limits[j], max_linear_limits[j]) for j in range(len(min_linear_limits))])
            new_transform.transform.rotation = Quaternion(*quaternion_from_euler(*[random.uniform(min_angular_limits[j], max_angular_limits[j]) for j in range(len(min_angular_limits))]))
            transformations.append(new_transform)

        return transformations

    def add_pose_goal(self, request, pose, eef_link):
        '''
        Add a pose goal to the given bio_ik request.
        '''
        goal = PoseGoal()
        goal.link_name = eef_link
        goal.weight = 20.0
        goal.pose = pose.pose
        request.pose_goals = [goal]
        return request

    def score(self, joint_state):
        '''
        Calculate the cost value for the given joint state.
        '''
        epsilon_old = math.pi

        # Calculations of delta for cost function
        delta = []
        epsilon = []
        joints = joint_state.name
        for i in range(len(joints)):
            # Ignore continuous joints
            if joints[i] in ["r_forearm_roll_joint", "l_forearm_roll_joint", "l_wrist_roll_joint"]:
                continue
            bounds = self.robot.get_joint(joints[i]).bounds()
            delta.append(min(abs(bounds[1] - joint_state.position[i]), abs(joint_state.position[i] - bounds[0])))
            epsilon.append(abs((bounds[1] - bounds[0])/2))

        # Implementation of cost function in bimanual functional regrasping paper with corresponding epsilons
        score = 0
        for i in range(len(delta)):
            score += 1/epsilon[i]**2 * delta[i]**2 - 2/epsilon[i] * delta[i] + 1
        score = score/len(delta)

        if self.debug and self.verbose:
            rospy.loginfo("Score with new epsilon: {}".format(score))

        return score

    def check_pose(self, pose, manipulator, robot_state):
        '''
        Get a joint configuration for the given pose and manipulator.
        '''
        # Set parameters based on manipulator and mode
        if manipulator == "hand":
            group_name = "right_arm"
            joints = self.hand.get_active_joints()
            robot_description = "/robot_description"
            eef = "rh_manipulator"
            goal_type = "fixed"
        elif manipulator == "gripper":
            group_name = "left_arm"
            joints = self.left_arm.get_active_joints()
            robot_description = "/robot_description"
            eef = "l_gripper_tool_frame"
            goal_type = "fixed"

        # Generate joint configuration through bio_ik
        request = prepare_bio_ik_request(group_name, robot_state, robot_description, timeout_seconds = 0.05)
        if goal_type == "rot_inv":
            request = self.add_can_goals(request, pose, eef)
        elif goal_type == "fixed":
            request = self.add_pose_goal(request, pose, eef)
        result = self.bio_ik_srv(request).ik_response
        # If result is not feasible, no further checking necessary
        if result.error_code.val != 1:
            if self.debug:
                if self.verbose:
                    rospy.loginfo("{} failed.".format(manipulator))
                display_state = DisplayRobotState()
                display_state.state.joint_state = result.solution.joint_state
                if manipulator == "hand":
                    self.debug_hand_state_pub.publish(display_state)
                elif manipulator == "gripper":
                    self.debug_gripper_state_pub.publish(display_state)
            return None, None
        else:
            filtered_joint_state = filter_joint_state(result.solution.joint_state, joints)
            return filtered_joint_state, result.solution_fitness

    def check_poses(self, gripper_pose, hand_pose):
        '''
        Get a joint configuration for both manipulators and check their validity.
        '''
        # To have the same initial state for both planning approaches
        initial_state = self.robot.get_current_state()

        # Get hand solution
        hand_state, hand_fitness = self.check_pose(hand_pose, "hand", initial_state)
        if hand_state is None:
            return 1, None, None

        # Apply hand_state to initial_state so the gripper sampling checks for collisions with the hand
        new_positions = list(initial_state.joint_state.position)
        for i in range(len(hand_state.name)):
            index = initial_state.joint_state.name.index(hand_state.name[i])
            new_positions[index] = hand_state.position[i]
        initial_state.joint_state.position = new_positions

        # Get gripper solution
        gripper_state, gripper_fitness = self.check_pose(gripper_pose, "gripper", initial_state)
        if gripper_state is None:
            return 1, None, None

        # Combine joint_states for combined score
        combined_joint_state = JointState()
        combined_joint_state.name = hand_state.name + gripper_state.name
        combined_joint_state.position = hand_state.position + gripper_state.position

        if self.debug and self.verbose:
            rospy.loginfo("Score hand: {}".format(self.score(hand_state)))
            rospy.loginfo("Score gripper: {}".format(self.score(gripper_state)))
            rospy.loginfo("Hand fitness: {}".format(hand_fitness))
            rospy.loginfo("Gripper fitness: {}".format(gripper_fitness))

        return self.score(combined_joint_state), hand_state, gripper_state

    def send_handover_frame(self, frame_pose):
        '''
        Set pose of the handover frame for current grasp pose.
        '''
        rospy.sleep(1)
        self.handover_frame_pub.publish(frame_pose)
        rospy.sleep(1)

    def get_handover_frame_pose(self, gripper_pose, hand_pose):
        '''
        Calculates the pose of the handover frame for the given gripper and hand poses.
        '''
        frame_pose = PoseStamped()
        frame_pose.header.frame_id = "base_footprint"
        frame_pose.pose.orientation.w = 1
        x = min(gripper_pose.pose.position.x, hand_pose.pose.position.x) + abs(gripper_pose.pose.position.x - hand_pose.pose.position.x)/2
        frame_pose.pose.position.x = x
        y = min(gripper_pose.pose.position.y, hand_pose.pose.position.y) + abs(gripper_pose.pose.position.y - hand_pose.pose.position.y)/2
        frame_pose.pose.position.y = y
        z = min(gripper_pose.pose.position.z, hand_pose.pose.position.z) + abs(gripper_pose.pose.position.z - hand_pose.pose.position.z)/2
        frame_pose.pose.position.z = z
        return frame_pose

    def write_float_array_to_bag(self, topic, array):
        '''
        Helper function to write float array into a rosbag.
        '''
        array_msg = Float32MultiArray()
        array_msg.data = array
        self.bag.write(topic, array_msg)

    def create_marker(self, name):
        '''
        Creates an empty RVIZ cube list marker.
        '''
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.ns = name
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.points = []
        marker.colors = []
        marker.scale = Vector3(0.01, 0.01, 0.01)
        return marker

    def get_handover_transform(self, gripper_pose, hand_pose):
        '''
        Get a suitable handover pose transform for the given hand and gripper poses.
        '''
        frame_pose = self.get_handover_frame_pose(gripper_pose, hand_pose)
        self.send_handover_frame(frame_pose)

        # Publish handover frame position as marker
        if self.debug:
            handover_marker = self.create_marker("handover_marker")
            translated_pose = frame_pose
            translated_pose.pose.position.x += 0.12
            translated_pose.pose.position.y += 0.06
            handover_marker.points.append(translated_pose.pose.position)
            handover_marker.colors.append(ColorRGBA(0, 0, 1, 1))
            self.debug_handover_marker_pub.publish(handover_marker)

        # Lookup transforms
        base_handover_transform = self.tf_buffer.lookup_transform("handover_frame", "base_footprint", rospy.Time(0))
        handover_base_transform = self.tf_buffer.lookup_transform("base_footprint", "handover_frame", rospy.Time(0))

        # Transform gripper and hand pose to handover_frame to apply transforms
        gripper_pose = do_transform_pose(gripper_pose, base_handover_transform)
        hand_pose = do_transform_pose(hand_pose, base_handover_transform)

        # Get the early stopping cost threshold and maximum pose distance for the IK solution
        if self.object_type == "can" or self.object_type == "bleach" or self.object_type == "roll":
            # From workspace analysis
            if self.side == "side":
                score_limit = 0.19
            else:
                score_limit = 0.16
        deviation_limit = 0.01

        # Initialize variables to find best transforms
        best_score = 1
        best_transform = None
        best_hand = None
        best_gripper = None

        # Get trhe transforms
        if self.handover_pose_mode == "random_sample":
            transformations = self.get_random_sample_transformations(1000)
        elif self.handover_pose_mode == "load_sample":
            transformations = self.load_transforms()

        # Initialize variables for debugging
        if self.debug:
            debug_counter = 0
            best_debug_index = None
            poses = PoseArray()
            poses.header.frame_id = "handover_frame"
            poses.poses = []
            initial_robot_state = self.robot.get_current_state()
            hand_marker = self.create_marker("hand_markers")
            gripper_marker = self.create_marker("gripper_markers")

        # Iterate through all transform until one falls below the stopping threshold
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
            score, hand_joint_state, gripper_joint_state = self.check_poses(transformed_gripper, transformed_hand)
            if score < 1:
                hand_pos_diff, hand_quat_diff = self.calculate_fk_diff(hand_joint_state, transformed_hand, "hand")
                gripper_pos_diff, gripper_quat_diff = self.calculate_fk_diff(gripper_joint_state, transformed_gripper, "gripper")
                if self.debug and self.verbose:
                    rospy.loginfo("Hand pos diff: {}".format(hand_pos_diff))
                    rospy.loginfo("Hand quat diff: {}".format(hand_quat_diff))
                    rospy.loginfo("Gripper pos diff: {}".format(gripper_pos_diff))
                    rospy.loginfo("Gripper quat diff: {}".format(gripper_quat_diff))
                    rospy.loginfo("---")
                if hand_pos_diff > deviation_limit or gripper_pos_diff > deviation_limit:
                    score = 1

            # Publish joint configurations as Display States
            if self.debug:
                self.debug_gripper_pose_pub.publish(transformed_gripper)
                self.debug_hand_pose_pub.publish(transformed_hand)
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
                    self.debug_gripper_state_pub.publish(display_state)
                if not hand_joint_state is None:
                    display_state = DisplayRobotState()
                    display_state.state = initial_robot_state
                    positions = list(display_state.state.joint_state.position)
                    for joint in hand_joint_state.name:
                        index = hand_joint_state.name.index(joint)
                        new_value = hand_joint_state.position[index]
                        display_index = display_state.state.joint_state.name.index(joint)
                        positions[display_index] = new_value
                    positions = tuple(positions)
                    display_state.state.joint_state.position = positions
                    self.debug_hand_state_pub.publish(display_state)
                hand_marker.points.append(transformed_hand.pose.position)
                gripper_marker.points.append(transformed_gripper.pose.position)
                if score < 1:
                    hand_marker.colors.append(ColorRGBA(0, 1, 0, 1))
                    gripper_marker.colors.append(ColorRGBA(0, 1, 0, 1))
                else:
                    hand_marker.colors.append(ColorRGBA(1, 0, 0, 1))
                    gripper_marker.colors.append(ColorRGBA(1, 0, 0, 1))

            # Update current best cost
            if score < best_score:
                best_score = score
                best_transform = transformation
                best_hand = hand_joint_state
                best_gripper = gripper_joint_state
                if self.debug:
                    best_debug_index = deepcopy(debug_counter)

            if self.debug:
                if self.verbose:
                    rospy.loginfo("Transform score: {}".format(score))
                    rospy.loginfo("---")
                debug_counter += 1

            # Stop if score already good enough
            if best_score < score_limit:
                break

        if self.debug:
            if self.verbose:
                rospy.loginfo("Best score: {}".format(best_score))
                rospy.loginfo("Best transform: {}".format(best_transform))
            self.debug_sampled_poses_pub.publish(poses)
            self.debug_hand_markers_pub.publish(hand_marker)
            self.debug_gripper_markers_pub.publish(gripper_marker)

        return best_transform, best_hand, best_gripper

    def calculate_fk_diff(self, joint_values, target_pose, chain_type):
        '''
        Calculate difference between the FK computation of the given joint values and the target pose.
        '''
        target = list(joint_values.position)
        target = [self.robot.get_joint("torso_lift_joint").value()] + target
        if chain_type == "hand":
            pose_fk = self.kdl_kin_hand.forward(target, base_link = "base_footprint", end_link = "rh_manipulator")
        elif chain_type == "gripper":
            pose_fk = self.kdl_kin_gripper.forward(target, base_link = "base_footprint", end_link = "l_gripper_tool_frame")

        if self.debug and self.verbose:
            rospy.loginfo("Pose type: {}".format(chain_type))
            rospy.loginfo("Target pose:")
            rospy.loginfo(target_pose)

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
        '''
        Set fingers into default, spread out configuration.
        '''
        # Values except for thumb taken from spread hand.
        joint_values = dict(rh_THJ4=1.13446, rh_LFJ4=-0.31699402670752413, rh_FFJ4=-0.23131151280059523, rh_MFJ4=0.008929532157657268, rh_RFJ4=-0.11378487918959583)
        self.fingers.set_joint_value_target(joint_values)
        self.fingers.go()

    def setup_fingers_together(self):
        '''
        Set fingers into default, parallel configuration.
        '''
        joint_values = dict(rh_THJ4=1.13446, rh_LFJ4=0, rh_FFJ4=0, rh_MFJ4=0, rh_RFJ4=0)
        self.fingers.set_joint_value_target(joint_values)
        self.fingers.go()

    def move_sampled_pose(self, hand_pose):
        '''
        Move the hand and gripper to the handover pose according to a sampled pose.
        '''
        # Pose from where to start sampling
        gripper_pose = self.get_gripper_pose()

        # Setup pre-hand pose
        pre_hand_pose = deepcopy(hand_pose)
        if self.side == "top":
            pre_hand_pose.pose.position.z += 0.08
        elif self.side == "side":
            pre_hand_pose.pose.position.y += -0.08

        if self.debug:
            self.debug_hand_pose_pub.publish(hand_pose)
            self.debug_gripper_pose_pub.publish(gripper_pose)

        # Get required transforms. Write down required time to get transform if desired.
        if self.write_time:
            self.time_file.write('Sampling type: {} \n'.format(self.handover_pose_mode))
            self.time_file.write('Object type: {} \n'.format(self.object_type))
            self.time_file.write('Pose type: {} \n'.format(self.grasp_pose_mode))
            self.time_file.write('Side: {} \n'.format(self.side))
            self.time_file.write('Sampling start time: {} \n'.format(rospy.Time.now()))
        handover_transform, hand_joint_state, gripper_joint_state = self.get_handover_transform(gripper_pose, hand_pose)
        if self.write_time:
            self.time_file.write('Sampling finish time: {} \n'.format(rospy.Time.now()))
            self.time_file.write('--- \n')
        base_handover_transform = self.tf_buffer.lookup_transform("handover_frame", "base_footprint", rospy.Time(0))
        handover_base_transform = self.tf_buffer.lookup_transform("base_footprint", "handover_frame", rospy.Time(0))

        if not handover_transform is None:
            rospy.loginfo("Handover pose found.")
        else:
            rospy.logerr("No handover pose found.")
            return False, None

        # Setup pre_hand_pose
        pre_hand_pose = do_transform_pose(pre_hand_pose, base_handover_transform)
        pre_hand_pose = do_transform_pose(pre_hand_pose, handover_transform)
        pre_hand_pose = do_transform_pose(pre_hand_pose, handover_base_transform)

        if self.debug:
            self.debug_gripper_pose_pub.publish(gripper_pose)

        # Move gripper to generated joint state
        self.left_arm.set_joint_value_target(gripper_joint_state)
        self.left_arm.go()

        if self.debug:
            self.debug_hand_pose_pub.publish(hand_pose)

        # Get hand approach solution
        request = prepare_bio_ik_request("right_arm", self.robot.get_current_state(), "/robot_description")
        request = self.add_pose_goal(request, pre_hand_pose, 'rh_manipulator')
        result = self.bio_ik_srv(request).ik_response
        if result.error_code.val != 1:
            rospy.logerr("Bio_ik planning request returned error code {}.".format(result.error_code.val))
            rospy.loginfo("The calculated approach hand joint configuration has not found a correct solution. Continuing with moving directly to the target pose.")
        else:
            filtered_joint_state_pre_hand = filter_joint_state(result.solution.joint_state, self.hand.get_active_joints())
            # Move hand to requested pre_hand_pose
            pre_hand_pos_diff, pre_hand_quat_diff = self.calculate_fk_diff(filtered_joint_state_pre_hand, pre_hand_pose, "hand")
            if pre_hand_pos_diff < 0.01:
                rospy.loginfo("Moving right arm to handover pose.")
                self.hand.set_joint_value_target(filtered_joint_state_pre_hand)
                self.hand.go()
            else:
                rospy.loginfo("The calculated approach hand joint configuration has not found a correct solution. Continuing with moving directly to the target pose.")

        # Display hand pose before moving towards it
        if self.debug:
            display_state = DisplayRobotState()
            display_state.state = self.robot.get_current_state()
            positions = list(display_state.state.joint_state.position)
            for joint in hand_joint_state.name:
                index = hand_joint_state.name.index(joint)
                new_value = hand_joint_state.position[index]
                display_index = display_state.state.joint_state.name.index(joint)
                positions[display_index] = new_value
            positions = tuple(positions)
            display_state.state.joint_state.position = positions
            self.debug_state_pub.publish(display_state)
            hand_pose = do_transform_pose(hand_pose, base_handover_transform)
            hand_pose = do_transform_pose(hand_pose, handover_transform)
            hand_pose = do_transform_pose(hand_pose, handover_base_transform)
            self.debug_hand_pose_pub.publish(hand_pose)

        # Move hand to generated joint state
        self.hand.set_joint_value_target(hand_joint_state)
        self.hand.go()

        # Print final hand pose relative to gripper
        if self.debug:
            base_gripper_transform = self.tf_buffer.lookup_transform("l_gripper_tool_frame", "base_footprint", rospy.Time(0))
            final_hand = self.hand.get_current_pose()
            final_hand = do_transform_pose(final_hand, base_gripper_transform)
            rospy.loginfo(final_hand)

        return True, handover_transform

    def move_fixed_pose(self, hand_pose):
        '''
        Move hand and gripper to handover pose according to a fixed pose.
        '''
        # Move gripper
        gripper_pose = self.get_gripper_pose()
        self.left_arm.set_pose_target(gripper_pose)
        self.left_arm.go()
        rospy.sleep(1)

        if self.debug:
            self.debug_hand_pose_pub.publish(hand_pose)

        # Move hand
        self.hand.set_pose_target(hand_pose)
        self.hand.go()

        return True, None

    def get_hand_pose_pc(self):
        '''
        Get hand pose according to the point cloud.
        '''
        hand_pose = self.get_gripper_pose()

        # Find max and min values of the pointcloud
        gen = pc2.read_points(self.pc, field_names = ("x", "y", "z"), skip_nans = True)
        max_point = [-100, -100, -100]
        min_point = [100, 100, 100]
        for point in gen:
            for x in range(len(point)):
                if point[x] > max_point[x]:
                    max_point[x] = point[x]
                if point[x] < min_point[x]:
                    min_point[x] = point[x]

        # Get hand pose based on object and grasp type
        if self.object_type == "can" or self.object_type == "bleach" or self.object_type == "roll":
            if self.side == "top":
                self.setup_fingers()
                hand_pose.pose.position.x = min_point[0] + math.dist([max_point[0]], [min_point[0]])/2 - 0.05
                hand_pose.pose.position.y = min_point[1] + math.dist([max_point[1]], [min_point[1]])/2 - 0.02
                hand_pose.pose.position.z = max_point[2] + 0.03
                hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(-1.5708, 3.14159, -1.5708))
                return hand_pose
            elif self.side == "side":
                self.setup_fingers_together()
                hand_pose.pose.position.x = min_point[0] + math.dist([max_point[0]], [min_point[0]])/2 + 0.07
                hand_pose.pose.position.y = min_point[1] - 0.02
                hand_pose.pose.position.z += 0.1
                hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, -1.5708, -1.5708))
                return hand_pose

    def get_hand_pose_fixed(self):
        '''
        Get fixed hand pose.
        '''
        hand_pose = self.get_gripper_pose()
        if self.object_type == "can" or self.object_type == "bleach" or self.object_type == "roll":
            if self.side == "top":
                self.setup_fingers()
                hand_pose.pose.position.x += -0.05
                hand_pose.pose.position.y += -0.02
                hand_pose.pose.position.z += 0.167
                hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(-1.5708, 3.14159, -1.5708))
                return hand_pose
            elif self.side == "side":
                self.setup_fingers_together()
                hand_pose.pose.position.x += 0.06
                hand_pose.pose.position.y += -0.055
                hand_pose.pose.position.z += 0.1
                hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, -1.5708, -1.5708))
                return hand_pose

    def get_gripper_pose(self):
        '''
        Get the fixed pose for the gripper.
        '''
        gripper_pose = PoseStamped()
        gripper_pose.header.frame_id = "base_footprint"
        gripper_pose.pose.position.x = 0.4753863391864514
        gripper_pose.pose.position.y = 0.03476345653124885
        gripper_pose.pose.position.z = 0.6746350873056409
        gripper_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -1.5708))
        return gripper_pose

if __name__ == "__main__":
    mover = HandoverMover()

