#!/usr/bin/env python3

import rospy
from moveit_commander import roscpp_initialize, roscpp_shutdown, MoveGroupCommander, RobotCommander
from std_msgs.msg import Bool
from bimanual_handover_msgs.srv import GraspTesterSrv
from geometry_msgs.msg import WrenchStamped, PoseStamped
from moveit_msgs.msg import DisplayRobotState
from copy import deepcopy
from bio_ik_msgs.msg import PoseGoal, IKRequest
from bio_ik_msgs.srv import GetIK
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose

class GraspTester():

    def __init__(self, debug = True):
        rospy.init_node('grasp_tester_node')
        roscpp_initialize('')
        rospy.on_shutdown(self.shutdown)

        # Setup force subscriber
        self.current_force = None
        force_sub = rospy.Subscriber('/ft/l_gripper_motor', WrenchStamped, self.update_force)

        # Setup commanders
        self.right_arm = MoveGroupCommander('right_arm', ns = "/")
        self.right_arm.get_current_pose() # To initiate state monitor: see moveit issue #2715
        self.robot = RobotCommander()

        # Setup services
        rospy.wait_for_service('/bio_ik/get_bio_ik')
        self.bio_ik_srv = rospy.ServiceProxy('/bio_ik/get_bio_ik', GetIK)

        # Setup tf listener
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer)

        # Debug
        self.debug = debug
        if self.debug:
            self.debug_ik_solution_pub = rospy.Publisher('debug/grasp_tester/ik_solution', DisplayRobotState, queue_size = 1, latch = True)
            self.debug_pub_current = rospy.Publisher('debug/grasp_tester/pre_cartesian', PoseStamped, queue_size = 1, latch = True)
            self.debug_pub_plan = rospy.Publisher('debug/grasp_tester/plan_cartesian', PoseStamped, queue_size = 1, latch = True)
            self.debug_snapshot_pub = rospy.Publisher('debug/grasp_tester/debug_snapshot', Bool, queue_size = 1, latch = True)
            self.debug_snapshot_pub.publish(False)

        # Start service
        rospy.Service('grasp_tester_srv', GraspTesterSrv, self.test_grasp)
        rospy.spin()

    def shutdown(self):
        roscpp_shutdown()

    def update_force(self, wrench):
        self.current_force_x = wrench.wrench.force.x
        self.current_force_y = wrench.wrench.force.y
        self.current_force_z = wrench.wrench.force.z

    def prepare_bio_ik_request(self, group_name, timeout_seconds = 1):
        '''
        Prepares an IKRequest for the bio ik solver.

        :param string group_name: Name of the move_group for which this request
            is planned.
        :param float64 timeout_seconds: Number of seconds after which the solver
            stops if no pose was found.

        :return IKRequest: The prepared IKRequest.
        '''
        request = IKRequest()
        request.group_name = group_name
        request.approximate = True
        request.timeout = rospy.Duration.from_sec(timeout_seconds)
        request.avoid_collisions = True
        request.robot_state = self.robot.get_current_state()
        return request

    def add_goals(self, request, pose):
        '''
        Adds the necessary goals to the provided IKRequest.

        :param IKRequest request: The IKRequest to which the goals should be
            added to.
        :param PoseStamped pose: The pose for the end-effector goal.

        :return IKRequest: The IKRequest with the added goals.
        '''
        eef_pose_goal = PoseGoal()
        eef_pose_goal.link_name = 'rh_palm'
        eef_pose_goal.weight = 10.0
        eef_pose_goal.pose = pose
        eef_pose_goal.rotation_scale = 1.0
        request.pose_goals = [eef_pose_goal]
        return request

    def filter_joint_state(self, joint_state, move_group):
        '''
        Filters the given joint state for only the joint specified in the given
        move group.

        :param JointState joint_state: The joint state to be filtered.
        :param MoveGroup move_group: The move group used as the filter.

        :return JointState: The filtered joint state.
        '''
        filtered_joint_state = joint_state
        joint_names = move_group.get_active_joints()
        indices = [filtered_joint_state.name.index(joint_name) for joint_name in joint_names]
        filtered_joint_state.name = joint_names
        filtered_joint_state.position = [joint_state.position[x] for x in indices]
        if joint_state.velocity:
            filtered_joint_state.velocity = [joint_state.velocity[x] for x in indices]
        if joint_state.effort:
            filtered_joint_state.effort = [joint_state.effort[x] for x in indices]
        return filtered_joint_state

    def move_bio_ik(self, target_pose):
        request = self.prepare_bio_ik_request('right_arm')
        request = self.add_goals(request, target_pose.pose)
        response = self.bio_ik_srv(request).ik_response
        if not response.error_code.val == 1:
            if self.debug:
                display_state = DisplayRobotState()
                display_state.state = response.solution
                self.debug_ik_solution_pub.publish(display_state)
            raise Exception("Bio_ik planning failed with error code {}.".format(response.error_code.val))
        filtered_joint_state = self.filter_joint_state(response.solution.joint_state, self.right_arm)
        self.right_arm.set_joint_value_target(filtered_joint_state)
        plan = self.right_arm.go()
        if not plan:
            raise Exception("Moving to pose \n {} \n failed. No path was found to the joint state \n {}.".format(current_pose, filtered_joint_state))

    def enforce_bounds(self, joint_values):
        joint_names = self.right_arm.get_active_joints()
        for i in range(len(joint_names)):
            joint_object = self.robot.get_joint(joint_names[i])
            bounds = joint_object.bounds()
            if joint_values[i] < bounds[0]:
                joint_values[i] = bounds[0]
                rospy.loginfo("{} set from {} to {}.".format(joint_names[i], joint_values[i], bounds[0]))
            elif joint_values[i] > bounds[1]:
                joint_values[i] = bounds[1]
                rospy.loginfo("{} set from {} to {}.".format(joint_names[i], joint_values[i], bounds[1]))
        return joint_values

    def test_grasp(self, req):
        if self.debug:
            self.debug_snapshot_pub.publish(True)
        if req.direction == "x":
            prev_ft = deepcopy(self.current_force_x)
        elif req.direction == "y":
            prev_ft = deepcopy(self.current_force_y)
        elif req.direction == "z":
            prev_ft = deepcopy(self.current_force_z)
        else:
            rospy.logerr("Unknown direction {} for grasp_tester. Currently only x, y and z are implemented.")
            return False
        #rospy.loginfo("Initial force value: {}".format(prev_ft))

        # Get current pose
        current_pose = self.right_arm.get_current_pose()
        old_joint_values = deepcopy(self.right_arm.get_current_joint_values())

        # Transform from base_footprint into l_gripper_tool_frame
        base_gripper_transform = self.tf_buffer.lookup_transform("l_gripper_tool_frame", "base_footprint", rospy.Time(0))
        transformed_pose = do_transform_pose(current_pose, base_gripper_transform)

        # Adjust pose in l_gripper_tool_frame
        transformed_pose.pose.position.z += 0.005
        if self.debug:
            self.debug_pub_plan.publish(transformed_pose)

        # Transform pose back to base_fooprint
        gripper_base_transform = self.tf_buffer.lookup_transform("base_footprint", "l_gripper_tool_frame", rospy.Time(0))
        goal_pose = do_transform_pose(transformed_pose, gripper_base_transform)

        # Move to goal pose
        self.move_bio_ik(goal_pose)

        if self.debug:
            self.debug_snapshot_pub.publish(False)
        if req.direction == 'x':
            cur_ft = deepcopy(self.current_force_x)
        elif req.direction == 'y':
            cur_ft = deepcopy(self.current_force_y)
        elif req.direction == 'z':
            cur_ft = deepcopy(self.current_force_z)
        #rospy.loginfo("Afterwards force value: {}".format(cur_ft))

        # Reset previous upwards movement
        try:
            self.right_arm.set_joint_value_target(old_joint_values)
            self.right_arm.go()
        except Exception as e:
            rospy.logerr("Encountered exception [ {} ] while moving to pre-test pose. Trying again with bounded joints.".format(e))
            old_joint_values = self.enforce_bounds(old_joint_values)
            try:
                self.right_arm.set_joint_value_target(old_joint_values)
                self.right_arm.go()
            except Exception as e:
                rospy.logerr("Encountered exception [ {} ] while moving to pre-test pose with bounded joints.".format(e))
                input("Please decide how to solve this issue and press Enter to continue.")

        print("Force diff: {}".format(abs(prev_ft - cur_ft)))
        if abs(prev_ft - cur_ft) >= 2:
            return True
        else:
            return False

if __name__ == "__main__":
    gt = GraspTester()
