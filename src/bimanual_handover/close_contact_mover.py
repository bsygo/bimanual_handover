#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sr_robot_msgs.msg import Biotac

class CloseContactMover():

    def __init__(self):
        self.joints = ['ffj2', 'mfj2', 'rfj2', 'thj5']
        self.joint_publishers = {}
        self.joint_publishers = [rospy.Publisher('sh_rh_' + joint + '_position_controller/command', Float64, queue_size = 1) for joint in self.joints]
        self.biotac_subs = [rospy.Subscriber('rh/tactile/biotac_' + joint, Biotac, callback = self.biotac_callback, callback_args = joint, queue_size = 10) for joint in self.joints]
        self.joint_state_sub = rospy.Subscriber('rh/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        self.current_biotac_values = [0, 0, 0, 0]
        self.current_joint_values = [0, 0, 0, 0]
        self.initial_biotac_values = [0, 0, 0, 0]
        self.biotac_threshold = 5 # Value taken from biotac manual
        self.wait_for_initial_values()

    def wait_for_initial_values(self):
        values_set = False
        while not values_set:
            values_set = 0 in self.current_biotac_values 
        self.initial_biotac_values = self.current_biotac_values

    def biotac_callback(self, sensor_data, joint_name):
        index = self.joints.index(joint_name)
        self.current_biotac_values[index] = sensor_data.electrodes

    def joint_callback(self, joint_state):
        indices = [joint_state.name.index('rh_FFJ2'), joint_state.name.index('rh_MFJ2'), joint_state.name.index('rh_RFJ2'), joint_state.name.index('rh_THJ5')]
        self.current_joint_values = [joint_state.position[x] for x in indices]

    def move_until_contacts(self):
        contacts = [False, False, False, False]
        finished = False
        targets = [Float64(1.57), Float64(1.57), Float64(1.57), Float64(1.0)]
        for i in range(len(self.joint_publishers)):
            self.joint_publishers[i].publish(targets[i])
        while not finished:
            for i in range(len(self.joints)):
                if not contacts[i]:
                    if (self.current_biotac_values[i] > self.initial_biotac_values[i] + self.biotac_threshold) or (self.current_biotac_values[i] < self.initial_biotac_values[i] - self.biotac_threshold):
                        contacts[i] = True
                        self.joint_publishers[i].publish(Float64(self.current_joint_values[i]))
            if sum(contacts) == 4:
                finished = True

if __name__ == '__main__':
    rospy.init_node('close_contact_mover')
    ccm = CloseContactMover()
    ccm.move_until_contacts()
