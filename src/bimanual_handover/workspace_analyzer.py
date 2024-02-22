#!/usr/bin/env python3

import rospy
from bimanual_handover.bio_ik_helper_functions import prepare_bio_ik_request, filter_joint_state
import json
from bio_ik_msgs.msg import IKRequest, PoseGoal
from bio_ik_msgs.srv import GetIK
from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion, PoseArray, Vector3, Point
from tf.transformations import quaternion_from_euler, quaternion_from_matrix, quaternion_multiply, quaternion_inverse
import math
import numpy as np
from sensor_msgs.msg import JointState
from datetime import datetime
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, RobotCommander
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from copy import deepcopy
from moveit_msgs.msg import DisplayRobotState
import rospkg

class StepTransform():

    def __init__(self, x, y, z):
        # Initialize coordinates
        self.x = x
        self.y = y
        self.z = z
        self.trans_step = 0.06
        self.rot_step = math.pi * 30/180
        self.rot_low_step = -3
        self.rot_up_step = 3

        # Initialize evaluation values
        self.number_solutions = None
        self.min_score = None
        self.avg_score = None
        self.scores = None
        self.gripper_positions = None
        self.hand_positions = None

    def get_transform_msgs(self):
        rotations = [[x, y, z] for x in range(self.rot_low_step, self.rot_up_step + 1) for y in range(self.rot_low_step, self.rot_up_step + 1) for z in range(self.rot_low_step, self.rot_up_step + 1)]
        transformations = []
        for rotation in rotations:
            new_transform = TransformStamped()
            new_transform.header.frame_id = "handover_frame"
            new_transform.child_frame_id = "handover_frame"
            new_transform.transform.translation = Vector3(*[self.x * self.trans_step, self.y * self.trans_step, self.z * self.trans_step])
            new_transform.transform.rotation = Quaternion(*quaternion_from_euler(*[rot_axis_steps * self.rot_step for rot_axis_steps in rotation]))
            transformations.append(new_transform)

        return transformations

    def is_evaluated(self):
        return self.number_solutions is None

    def set_eval_results(self, number_solutions, min_score, avg_score, scores, gripper_positions, hand_positions):
        self.number_solutions = number_solutions
        self.min_score = min_score
        self.avg_score = avg_score
        self.scores = scores
        self.gripper_positions = gripper_positions
        self.hand_positions = hand_positions

    def key(self):
        return str([self.x, self.y, self.z])

    def serialize(self):
        serialized_gripper_positions = [[p.x, p.y, p.z] for p in self.gripper_positions]
        serialized_hand_positions = [[p.x, p.y, p.z] for p in self.hand_positions]
        return {"x": int(self.x), "y": int(self.y), "z": int(self.z), "number_solutions": int(self.number_solutions), "min_score": float(self.min_score), "avg_score": float(self.avg_score), "scores": self.scores, "gripper_positions": serialized_gripper_positions, "hand_positions": serialized_hand_positions}

    def parse(self, data):
        self.x = data["x"]
        self.y = data["y"]
        self.z = data["z"]
        self.number_solutions = data["number_solutions"]
        self.min_score = data["min_score"]
        self.avg_score = data["avg_score"]
        self.scores = data["scores"]
        serialized_gripper_points = data["gripper_positions"]
        gripper_points = [Point(*point) for point in serialized_gripper_points]
        self.gripper_positions = gripper_points
        serialized_hand_points = data["hand_positions"]
        hand_points = [Point(*point) for point in serialized_hand_points]
        self.hand_positions = hand_points
        return self

class TransformHandler():

    def __init__(self):
        self.wa = WorkspaceAnalyzer()
        self.data = {}
        self.time = datetime.now().strftime("%d_%m_%Y_%H_%M")
        self.pkg_path = rospkg.RosPack().get_path('bimanual_handover')

    def get_data(self):
        return self.data

    def initial_block(self):
        x = np.arange(0, 1)
        y = np.arange(0, 1)
        z = np.arange(0, 1)
        x, y, z = np.meshgrid(x, y, z)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        for step in grid:
            step_transform = StepTransform(step[0], step[1], step[2])
            self.data[step_transform.key()] = step_transform
            self.wa.analyze_transform(step_transform)
            self.save()

    def get_border_values(self):
        border_dict = {"min_x": 1000, "min_y": 1000, "min_z": 1000, "max_x": -1000, "max_y": -1000, "max_z": -1000}
        for transform in self.data.values():
            if transform.x < border_dict["min_x"]:
                border_dict["min_x"] = transform.x
            if transform.y < border_dict["min_y"]:
                border_dict["min_y"] = transform.y
            if transform.z < border_dict["min_z"]:
                border_dict["min_z"] = transform.z
            if transform.x > border_dict["max_x"]:
                border_dict["max_x"] = transform.x
            if transform.y > border_dict["max_y"]:
                border_dict["max_y"] = transform.y
            if transform.z > border_dict["max_z"]:
                border_dict["max_z"] = transform.z
        return border_dict

    def expand(self):
        expand = False
        limit = 0 # 309
        border_dict = self.get_border_values()
        expansion_grid_x = [border_dict["min_x"], border_dict["max_x"]]
        expansion_grid_y = [border_dict["min_y"], border_dict["max_y"]]
        expansion_grid_z = [border_dict["min_z"], border_dict["max_z"]]
        for transform in self.data.values():
            if transform.x == border_dict["min_x"]:
                if transform.number_solutions <= limit:
                    expansion_grid_x[0] = border_dict["min_x"] - 1
                    expand = True
            if transform.y == border_dict["min_y"]:
                if transform.number_solutions <= limit:
                    expansion_grid_y[0] = border_dict["min_y"] - 1
                    expand = True
            if transform.z == border_dict["min_z"]:
                if transform.number_solutions <= limit:
                    expansion_grid_z[0] = border_dict["min_z"] - 1
                    expand = True
            if transform.x == border_dict["max_x"]:
                if transform.number_solutions <= limit:
                    expansion_grid_x[1] = border_dict["max_x"] + 1
                    expand = True
            if transform.y == border_dict["max_y"]:
                if transform.number_solutions <= limit:
                    expansion_grid_y[1] = border_dict["max_y"] + 1
                    expand = True
            if transform.z == border_dict["max_z"]:
                if transform.number_solutions <= limit:
                    expansion_grid_z[1] = border_dict["max_z"] + 1
                    expand = True

        x = np.arange(expansion_grid_x[0], expansion_grid_x[1] + 1)
        y = np.arange(expansion_grid_y[0], expansion_grid_y[1] + 1)
        z = np.arange(expansion_grid_z[0], expansion_grid_z[1] + 1)
        x, y, z = np.meshgrid(x, y, z)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        skip_counter = 0
        for step in grid:
            if not str(step.tolist()) in self.data.keys():
                step_transform = StepTransform(step[0], step[1], step[2])
                self.data[step_transform.key()] = step_transform
                self.wa.analyze_transform(step_transform)
                self.save()
            else:
                skip_counter += 1
        rospy.loginfo("{} values were skipped during last expansion.".format(skip_counter))
        return expand

    def save(self):
        serialize_data = {}
        for key, value in self.data.items():
            serialize_data[key] = value.serialize()
        with open(self.pkg_path + "/data/workspace_analysis/" + "workspace_analysis_{}.json".format(self.time), "w") as file:
            json.dump(serialize_data, file)

    def load(self, filename):
        with open(self.pkg_path + "/data/workspace_analysis/" + filename, "r") as file:
            serialized_data = json.load(file)
        for key, value in serialized_data.items():
            self.data[key] = StepTransform(0, 0, 0).parse(value)

    def save_independent(data, filepath):
        serialize_data = {}
        for key, value in data.items():
            serialize_data[key] = value.serialize()
        with open(filepath, "w") as file:
            json.dump(serialize_data, file)

    def load_independent(filename):
        data = {}
        with open(filename, "r") as file:
            serialized_data = json.load(file)
        for key, value in serialized_data.items():
            data[key] = StepTransform(0, 0, 0).parse(value)
        return data

class WorkspaceAnalyzer():

    def __init__(self):
        roscpp_initialize('')
        rospy.on_shutdown(self.shutdown)

        # Setup commanders
        self.hand = MoveGroupCommander("right_arm", ns = "/")
        self.hand.get_current_pose() # To initiate state monitor: see moveit issue #2715
        self.left_arm = MoveGroupCommander("left_arm", ns = "/")
        self.fingers = MoveGroupCommander("right_fingers", ns = "/")
        self.robot = RobotCommander()

        rospy.wait_for_service('/bio_ik/get_bio_ik')
        self.bio_ik_srv = rospy.ServiceProxy('/bio_ik/get_bio_ik', GetIK)

        self.side = "side"
        self.object_type = "can"
        self.verbose = rospy.get_param("/handover/handover_mover/verbose")

        # Setup FK
        self.fk_robot = URDF.from_parameter_server(key = "/robot_description")
        self.kdl_kin_hand = KDLKinematics(self.fk_robot, "base_footprint", "rh_manipulator")
        self.kdl_kin_gripper = KDLKinematics(self.fk_robot, "base_footprint", "l_gripper_tool_frame")

        # Setup transform listener and publisher
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer)
        self.handover_frame_pub = rospy.Publisher("handover_frame_pose", PoseStamped, queue_size = 1)

        self.debug = True
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

    def shutdown(self):
        roscpp_shutdown()

    def add_pose_goal(self, request, pose, eef_link):
        goal = PoseGoal()
        goal.link_name = eef_link
        goal.weight = 20.0
        goal.pose = pose.pose
        request.pose_goals = [goal]
        return request

    def score(self, joint_state):
        # Should be the maximum distance to any joint limit. Need to check joint limits if correct
        epsilon_old = math.pi

        # Calculations of delta for cost function
        delta = []
        epsilon = []
        joints = joint_state.name
        for i in range(len(joints)):
            # Skip continuous joints
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

        if self.verbose:
            rospy.loginfo("Score: {}".format(score))

        return score

    def check_pose(self, pose, manipulator, robot_state):
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

        if self.verbose:
            rospy.loginfo("Score hand: {}".format(self.score(hand_state)))
            rospy.loginfo("Score gripper: {}".format(self.score(gripper_state)))
            rospy.loginfo("Hand fitness: {}".format(hand_fitness))
            rospy.loginfo("Gripper fitness: {}".format(gripper_fitness))

        return self.score(combined_joint_state), hand_state, gripper_state

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

    def create_marker(self, name):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.ns = name
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.points = []
        marker.colors = []
        marker.scale = Vector3(0.06, 0.06, 0.06)
        return marker

    def create_display_state(self, initial_state, changed_state):
        display_state = DisplayRobotState()
        display_state.state = initial_state
        positions = list(display_state.state.joint_state.position)
        for joint in changed_state.name:
            index = changed_state.name.index(joint)
            new_value = changed_state.position[index]
            display_index = display_state.state.joint_state.name.index(joint)
            positions[display_index] = new_value
        positions = tuple(positions)
        display_state.state.joint_state.position = positions
        return display_state

    def analyze_transform(self, pos_transform):
        gripper_pose = self.get_gripper_pose()
        hand_pose = self.get_hand_pose_fixed()
        self.send_handover_frame(gripper_pose, hand_pose)

        # Lookup transforms
        base_handover_transform = self.tf_buffer.lookup_transform("handover_frame", "base_footprint", rospy.Time(0))
        handover_base_transform = self.tf_buffer.lookup_transform("base_footprint", "handover_frame", rospy.Time(0))

        # Transform gripper and hand pose to handover_frame to apply transforms
        gripper_pose = do_transform_pose(gripper_pose, base_handover_transform)
        hand_pose = do_transform_pose(hand_pose, base_handover_transform)

        # Setup for iterating through transformations
        deviation_limit = 0.01
        initial_robot_state = self.robot.get_current_state()

        # Setup data collection
        if self.debug:
            poses = PoseArray()
            poses.header.frame_id = "handover_frame"
            poses.poses = []
        iteration_results = []
        iteration_scores = []
        gripper_positions = []
        hand_positions = []
        hand_marker = self.create_marker("hand_markers")
        gripper_marker = self.create_marker("gripper_markers")

        transformations = pos_transform.get_transform_msgs()

        # Iterate through transforms
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
                if self.verbose:
                    rospy.loginfo("Hand pos diff: {}".format(hand_pos_diff))
                    rospy.loginfo("Gripper pos diff: {}".format(gripper_pos_diff))
                if (hand_pos_diff or gripper_pos_diff) > deviation_limit:
                    score = 1

            # Debug hand and gripper states
            if self.debug:
                self.debug_gripper_pose_pub.publish(transformed_gripper)
                self.debug_hand_pose_pub.publish(transformed_hand)
                if not gripper_joint_state is None:
                    display_state = self.create_display_state(initial_robot_state, gripper_joint_state)
                    self.debug_gripper_state_pub.publish(display_state)
                if not hand_joint_state is None:
                    display_state = self.create_display_state(initial_robot_state, hand_joint_state)
                    self.debug_hand_state_pub.publish(display_state)

            # Add points and colors for hand and gripper markers
            hand_positions.append(transformed_hand.pose.position)
            gripper_positions.append(transformed_gripper.pose.position)
            hand_marker.points.append(transformed_hand.pose.position)
            gripper_marker.points.append(transformed_gripper.pose.position)
            if score < 1:
                hand_marker.colors.append(ColorRGBA(0, 1, 0, 1))
                gripper_marker.colors.append(ColorRGBA(0, 1, 0, 1))
            else:
                hand_marker.colors.append(ColorRGBA(1, 0, 0, 1))
                gripper_marker.colors.append(ColorRGBA(1, 0, 0, 1))

            # Add transform to iteration results
            iteration_scores.append(score)
            if score == 1:
                iteration_results.append(1)
            else:
                iteration_results.append(0)

            if self.verbose:
                rospy.loginfo("Transform score: {}".format(score))
                rospy.loginfo("---")

        if self.debug:
            self.debug_sampled_poses_pub.publish(poses)
            self.debug_hand_markers_pub.publish(hand_marker)
            self.debug_gripper_markers_pub.publish(gripper_marker)

        # Add results of last iteration
        number_results = sum(iteration_results)
        min_score = min(iteration_scores)
        avg_score = sum(iteration_scores)/len(iteration_scores)

        pos_transform.set_eval_results(number_results, min_score, avg_score, iteration_results, gripper_positions, hand_positions)

    def calculate_fk_diff(self, joint_values, target_pose, chain_type):
        target = list(joint_values.position)
        target = [self.robot.get_joint("torso_lift_joint").value()] + target
        if chain_type == "hand":
            pose_fk = self.kdl_kin_hand.forward(target, base_link = "base_footprint", end_link = "rh_manipulator")
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

    def setup_fingers_together(self):
        joint_values = dict(rh_THJ4=1.13446, rh_LFJ4=0, rh_FFJ4=0, rh_MFJ4=0, rh_RFJ4=0)
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

    def get_hand_pose_fixed(self):
        hand_pose = self.get_gripper_pose()
        if self.object_type == "can":
            if self.side == "top":
                self.setup_fingers()
                hand_pose.pose.position.x += 0.0
                hand_pose.pose.position.y += -0.04
                hand_pose.pose.position.z += 0.167
                hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(-1.5708, 3.14159, 0))
                return hand_pose
            elif self.side == "side":
                self.setup_fingers_together()
                hand_pose.pose.position.x += 0.04
                hand_pose.pose.position.y += -0.055
                hand_pose.pose.position.z += 0.1
                hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, -1.5708, -1.5708))
                return hand_pose
        elif self.object_type == "book":
            if self.side == "top":
                self.setup_fingers()
                hand_pose.pose.position.x += -0.05
                hand_pose.pose.position.y += -0.02
                hand_pose.pose.position.z += 0.167
                hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(-1.5708, 3.14159, -1.5708))
                return hand_pose
            # Not correct numbers
            elif self.side == "side":
                self.setup_fingers_together()
                hand_pose.pose.position.x += 0.03
                hand_pose.pose.position.y += -0.13
                hand_pose.pose.position.z += 0.09
                hand_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, -1.5708, -1.5708))
                return hand_pose

def main():
    load = "workspace_analysis_20_02_2024_16_17.json"
    rospy.init_node("workspace_analyzer")
    #load = None
    th = TransformHandler()
    rospy.loginfo("TransformHandler started.")
    if load is None:
        rospy.loginfo("Now load file provided. Initialized without previous values.")
        th.initial_block()
    else:
        rospy.loginfo("Loading data from file {}.".format(load))
        th.load(load)
    expanded = True
    while expanded:
        rospy.loginfo("Workspace expanded.")
        expanded = th.expand()
    rospy.loginfo("Workspace expansion finished.")

if __name__ == "__main__":
    main()
