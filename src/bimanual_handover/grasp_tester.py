#!/usr/bin/env python3

import rospy
from moveit_commander import roscpp_initialize, roscpp_shutdown, MoveGroupCommander, RobotCommander
from bimanual_handover_msgs.srv import GraspTesterSrv
from geometry_msgs.msg import WrenchStamped, PoseStamped
from copy import deepcopy
from bio_ik_msgs.msg import PoseGoal, IKRequest
from bio_ik_msgs.srv import GetIK

class GraspTester():

    def __init__(self):
        rospy.init_node('grasp_tester')
        roscpp_initialize('')
        rospy.on_shutdown(self.shutdown)
        force_sub = rospy.Subscriber('/ft/l_gripper_motor', WrenchStamped, self.update_force)
        rospy.wait_for_service('/bio_ik/get_bio_ik')
        self.bio_ik_srv = rospy.ServiceProxy('/bio_ik/get_bio_ik', GetIK)
        self.current_force = None
        self.right_arm = MoveGroupCommander('right_arm', ns = "/")
        self.right_arm.get_current_pose() # To initiate state monitor: see moveit issue #2715
        self.robot = RobotCommander()
        rospy.Service('grasp_tester', GraspTesterSrv, self.test_grasp)
        self.debug_pub_current = rospy.Publisher('debug/pre_cartesian', PoseStamped, queue_size = 1, latch = True)
        self.debug_pub_plan = rospy.Publisher('debug/plan_cartesian', PoseStamped, queue_size = 1, latch = True)
        rospy.spin()

    def shutdown(self):
        roscpp_shutdown()

    def update_force(self, wrench):
        self.current_force = wrench.wrench.force.y

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

    def test_grasp(self, req):
        prev_ft = deepcopy(self.current_force)
        rospy.loginfo("Initial force value: {}".format(prev_ft))

        # For some reason, this needs to happen twice or otherwise the orientation is messed up
        #current_pose = self.right_arm.get_current_pose()
        #rospy.sleep(1)
        current_pose = self.right_arm.get_current_pose()

        request = self.prepare_bio_ik_request('right_arm')
        current_pose.pose.position.z += 0.005
        self.debug_pub_plan.publish(current_pose)
        request = self.add_goals(request, current_pose.pose)
        response = self.bio_ik_srv(request).ik_response
        if not response.error_code.val == 1:
            raise Exception("Bio_ik planning failed with error code {}.".format(response.error_code.val))
        filtered_joint_state = self.filter_joint_state(response.solution.joint_state, self.right_arm)
        self.right_arm.set_joint_value_target(filtered_joint_state)
        plan = self.right_arm.go()
        if not plan:
            raise Exception("Moving to pose \n {} \n failed. No path was found to the joint state \n {}.".format(current_pose, filtered_joint_state))

        cur_ft = deepcopy(self.current_force)
        rospy.loginfo("Afterwards force value: {}".format(cur_ft))
        if prev_ft + 2 <= cur_ft:
            return True
        else:
            return False

if __name__ == "__main__":
    gt = GraspTester()
