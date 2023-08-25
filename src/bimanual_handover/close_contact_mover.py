#!/usr/bin/env python3

import rospy
import rospkg
from datetime import datetime
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_msgs.msg import BiotacAll
from bimanual_handover.srv import CCM
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from copy import deepcopy
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from sr_robot_msgs.msg import BiotacAll
from pr2_msgs.msg import PressureState
from geometry_msgs.msg import WrenchStamped

class CloseContactMover():

    def __init__(self, debug = False, collect = False):
        rospy.init_node('close_contact_mover')
        self.joints_dict = {'rh_FFJ2': 0, 'rh_FFJ3': 0, 'rh_MFJ2': 1, 'rh_MFJ3': 1, 'rh_RFJ2': 2, 'rh_RFJ3': 2, 'rh_LFJ2': 3, 'rh_LFJ3': 3, 'rh_THJ5': 4}
        self.joints = list(self.joints_dict.keys())
        self.joint_client = actionlib.SimpleActionClient('/hand/rh_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.biotac_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, callback = self.biotac_callback, queue_size = 10)
        self.joint_state_sub = rospy.Subscriber('/hand/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        self.current_biotac_values = [0, 0, 0, 0, 0]
        self.current_joint_values = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.current_joint_state = None
        self.initial_biotac_values = [0, 0, 0, 0, 0]
        self.biotac_threshold = 20 # Value taken from biotac manual
        self.debug = debug
        self.collect = collect
        if self.debug:
            self.debug_pub = rospy.Publisher('debug/ccm', DisplayTrajectory, latch = True, queue_size = 1)
        if self.collect:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('bimanual_handover')
            now = datetime.now()
            current_date = now.strftime("%d-%m-%Y:%H-%M-%S")
            name = "closing_attempt_" + current_date + ".txt"
            self.data_file = open(pkg_path + "/data/" + name, "w")
        self.joint_client.wait_for_server()
        rospy.Service('ccm', CCM, self.move_until_contacts)
        rospy.spin()

    def wait_for_initial_values(self):
        values_set = False
        while not values_set:
            values_set = 0 not in self.current_biotac_values
        self.initial_biotac_values = deepcopy(self.current_biotac_values)
        while self.current_joint_state is None:
            pass

    def biotac_callback(self, sensor_data):
        self.current_biotac_values[0] = sensor_data.tactiles[0].electrodes
        self.current_biotac_values[1] = sensor_data.tactiles[1].electrodes
        self.current_biotac_values[2] = sensor_data.tactiles[2].electrodes
        modified_data = sensor_data.tactiles[3].electrodes[:6] + sensor_data.tactiles[3].electrodes[10:15] + sensor_data.tactiles[3].electrodes[16:] # volatile biotac sensors
        self.current_biotac_values[3] = modified_data
        self.current_biotac_values[4] = sensor_data.tactiles[4].electrodes

    def joint_callback(self, joint_state):
        indices = [joint_state.name.index(joint_name) for joint_name in self.joints]#joint_state.name.index('rh_FFJ2'), joint_state.name.index('rh_MFJ2'), joint_state.name.index('rh_RFJ2'), joint_state.name.index('rh_THJ5')]
        self.current_joint_values = [joint_state.position[x] for x in indices]
        self.current_joint_state = joint_state

    def create_joint_trajectory_msg(self, joint_names, targets):
        msg = JointTrajectory()
        msg.joint_names = joint_names
        point_msgs = []
        if not type(targets[0]) is list:
            for i in range(len(targets)):
                targets[i] = [targets[i]]
        for i in range(len(targets[0])):
            point_msg = JointTrajectoryPoint()
            point_msg.time_from_start = rospy.Duration(1)
            point_msg.positions = [targets[j][i] for j in range(len(targets))]
            point_msgs.append(point_msg)
        msg.points = point_msgs
        return msg

    def create_joint_trajectory_goal_msg(self, joint_names, targets):
        traj_msg = self.create_joint_trajectory_msg(joint_names, targets)
        goal_msg = FollowJointTrajectoryGoal(trajectory=traj_msg)
        return goal_msg

    def move_until_contacts(self, req):
        rospy.sleep(2)
        self.wait_for_initial_values()
        rospy.loginfo(self.initial_biotac_values[2])
        rospy.loginfo(self.initial_biotac_values[3])
        contacts = [False, False, False, False, False]
        targets = [1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.0]
        steps = []
        for x in range(len(targets)):
            diff = targets[x] - self.current_joint_values[x]
            steps_temp = [self.current_joint_values[x] + y * diff/30 for y in range(30)]
            steps.append(steps_temp)

        if self.debug:
            # Publish trajectory for debugging
            debug_traj = DisplayTrajectory()
            debug_traj.trajectory = [RobotTrajectory()]
            debug_traj.trajectory[0].joint_trajectory = self.create_joint_trajectory_msg(self.joints, steps)
            debug_traj.trajectory_start.joint_state = rospy.wait_for_message('/joint_states', JointState)
            self.debug_pub.publish(debug_traj)

        for x in range(len(steps[0])):
            if self.collect:
                self.data_file.write("Observation: ")
                pressure = rospy.wait_for_message('/pressure/l_gripper_motor', PressureState)
                tactile = rospy.wait_for_message('/hand/rh/tactile', BiotacAll)
                force = rospy.wait_for_message('/ft/l_gripper_motor', WrenchStamped)
                joints = rospy.wait_for_message('/joint_state', JointState)
                self.data_file.write(pressure)
                self.data_file.write(" ")
                self.data_file.write(tactile)
                self.data_file.write(" ")
                self.data_file.write(force)
                self.data_file.write(" ")
                self.data_file.write(joints)
                self.data_file.write(" ")
            used_joints = []
            used_steps = []
            for i in range(len(self.joints)):
                if not contacts[self.joints_dict[self.joints[i]]]:
                    used_joints.append(self.joints[i])
                    used_steps.append(steps[i][x])
            msg = self.create_joint_trajectory_goal_msg(used_joints, used_steps)
            self.joint_client.send_goal(msg)
            self.joint_client.wait_for_result()
            if not self.joint_client.get_result().error_code == 0:
                rospy.loginfo(self.joint_client.get_result().error_code)
            for i in range(len(contacts)):
                if not contacts[i]:
                    for j in range(len(self.current_biotac_values[i])):
                        if (self.current_biotac_values[i][j] > self.initial_biotac_values[i][j] + self.biotac_threshold) or (self.current_biotac_values[i][j] < self.initial_biotac_values[i][j] - self.biotac_threshold):
                            contacts[i] = True
                            rospy.loginfo('Contact found with joint {}.'.format(i))
                            if i == 3:
                                rospy.loginfo('Index: {}, Initial: {}, Contact: {}'.format(j, self.current_biotac_values[i][j], self.initial_biotac_values[i][j]))
                            break
            if self.collect:
                self.data_file.write("Result:  ")
                joints = rospy.wait_for_message('/joint_state', JointState)
                self.data_file.write("\n")
            if sum(contacts) == 5:
                print('contacts reached')
                return True
        if self.collect:
            self.data_file.close()
        print(contacts)
        return False

if __name__ == '__main__':
    ccm = CloseContactMover()
