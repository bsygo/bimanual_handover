#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_msgs.msg import BiotacAll
from bimanual_handover.srv import CCM
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from copy import deepcopy

class CloseContactMover():

    def __init__(self):
        self.joints = ['rh_FFJ2', 'rh_MFJ2', 'rh_RFJ2', 'rh_THJ5']
        #self.joint_publisher = rospy.Publisher('/hand/rh_trajectory_controller/command', JointTrajectory, queue_size = 10)
        self.joint_client = actionlib.SimpleActionClient('/hand/rh_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.biotac_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, callback = self.biotac_callback, queue_size = 10)
        self.joint_state_sub = rospy.Subscriber('/hand/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        self.current_biotac_values = [0, 0, 0, 0]
        self.current_joint_values = [0, 0, 0, 0]
        self.current_joint_state = None
        self.initial_biotac_values = [0, 0, 0, 0]
        self.biotac_threshold = 5 # Value taken from biotac manual
        self.wait_for_initial_values()
        self.joint_client.wait_for_server()
        rospy.Service('handover/ccm', CCM, self.move_until_contacts)
        rospy.spin()

    def wait_for_initial_values(self):
        values_set = False
        while not values_set:
            values_set = 0 not in self.current_biotac_values 
        print(values_set)
        print(self.current_biotac_values)
        self.initial_biotac_values = deepcopy(self.current_biotac_values)
        while self.current_joint_state is None:
            pass

    def biotac_callback(self, sensor_data):
        self.current_biotac_values[0] = sensor_data.tactiles[0].electrodes
        self.current_biotac_values[1] = sensor_data.tactiles[1].electrodes
        self.current_biotac_values[2] = sensor_data.tactiles[2].electrodes
        self.current_biotac_values[3] = sensor_data.tactiles[4].electrodes

    def joint_callback(self, joint_state):
        indices = [joint_state.name.index('rh_FFJ2'), joint_state.name.index('rh_MFJ2'), joint_state.name.index('rh_RFJ2'), joint_state.name.index('rh_THJ5')]
        self.current_joint_values = [joint_state.position[x] for x in indices]
        self.current_joint_state = joint_state

    def create_joint_trajectory_msg(self, joint_names, targets):
        msg = JointTrajectory()
        msg.joint_names = joint_names
        point_msg = JointTrajectoryPoint()
        point_msg.time_from_start = rospy.Duration(1)
        point_msg.positions = targets
        msg.points = [point_msg]
        return msg

    def create_joint_trajectory_goal_msg(self, joint_names, targets):
        traj_msg = self.create_joint_trajectory_msg(joint_names, targets)
        goal_msg = FollowJointTrajectoryGoal(trajectory=traj_msg)
        return goal_msg

    def move_until_contacts(self, req):
        print('start')
        contacts = [False, False, False, False]
        targets = [1.57, 1.57, 1.57, 1.0]
        steps = []
        for x in range(len(targets)):
            diff = targets[x] - self.current_joint_values[x]
            steps_temp = [self.current_joint_values[x] + y * diff/30 for y in range(30)]
            steps.append(steps_temp)
        for x in range(len(steps[0])):
            used_joints = []
            used_steps = []
            for i in range(len(self.joints)):
                if not contacts[i]:
                    used_joints.append(self.joints[i])
                    used_steps.append(steps[i][x])
            msg = self.create_joint_trajectory_goal_msg(used_joints, used_steps)
            # msg.header.stamp = rospy.Time.now()
            # self.joint_publisher.publish(msg)
            self.joint_client.send_goal(msg)
            self.joint_client.wait_for_result()
            if not self.joint_client.get_result().error_code == 0:
                rospy.loginfo(self.joint_client.get_result().error_code)
            # wait until movement finished -> convert to action client?
            print('---')
            print(self.current_biotac_values[0][2])
            print(self.initial_biotac_values[0][2])
            for i in range(len(self.joints)):
                if not contacts[i]:
                    for j in range(len(self.current_biotac_values[i])):
                        if (self.current_biotac_values[i][j] > self.initial_biotac_values[i][j] + self.biotac_threshold) or (self.current_biotac_values[i][j] < self.initial_biotac_values[i][j] - self.biotac_threshold):
                            contacts[i] = True
                            rospy.loginfo('Contact found with joint {}.'.format(self.joints[i]))
                            break
            if sum(contacts) == 4:
                print('contacts reached')
                return True
        print(contacts)
        return False

if __name__ == '__main__':
    rospy.init_node('close_contact_mover')
    ccm = CloseContactMover()
    #ccm.move_until_contacts()
