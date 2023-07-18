#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_msgs.msg import BiotacAll

class CloseContactMover():

    def __init__(self):
        self.joints = ['rh_FFJ2', 'rh_MFJ2', 'rh_RFJ2', 'rh_THJ5']
        self.joint_publisher = rospy.Publisher('/hand/rh_trajectory_controller/command', JointTrajectory, queue_size = 10)
        self.biotac_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, callback = self.biotac_callback, queue_size = 10)
        self.joint_state_sub = rospy.Subscriber('/hand/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        self.current_biotac_values = [0, 0, 0, 0]
        self.current_joint_values = [0, 0, 0, 0]
        self.current_joint_state = None
        self.initial_biotac_values = [0, 0, 0, 0]
        self.biotac_threshold = 5 # Value taken from biotac manual
        self.wait_for_initial_values()

    def wait_for_initial_values(self):
        values_set = False
        while not values_set:
            values_set = 0 not in self.current_biotac_values 
        print(values_set)
        print(self.current_biotac_values)
        self.initial_biotac_values = self.current_biotac_values
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

    def move_until_contacts(self):
        print('start')
        contacts = [False, False, False, False]
        finished = False
        targets = [1.57, 1.57, 1.57, 1.0]
        for i in range(len(self.joint_names)):
            msg = self.create_joint_trajectory_msg([self.joint_names[i]], [targets[i]])
            msg.header.stamp = rospy.Time.now()
            self.joint_publisher.publish(msg)
        print('targets send')
        while not finished:
            for i in range(len(self.joints)):
                if not contacts[i]:
                    for j in range(len(self.current_biotac_values[i])):
                        if (self.current_biotac_values[i][j] > self.initial_biotac_values[i][j] + self.biotac_threshold) or (self.current_biotac_values[i][j] < self.initial_biotac_values[i][j] - self.biotac_threshold):
                            contacts[i] = True
                            print('contact {}', i)
                            msg = self.create_joint_trajectory_msg()
                            msg.points[0].positions[i] = self.current_joint_values[i]
                            self.joint_publisher.publish(msg)
                            break
            if sum(contacts) == 4:
                finished = True

if __name__ == '__main__':
    rospy.init_node('close_contact_mover')
    ccm = CloseContactMover()
    ccm.move_until_contacts()
