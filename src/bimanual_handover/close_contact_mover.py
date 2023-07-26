#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_msgs.msg import BiotacAll
from bimanual_handover.srv import CCM
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory

class CloseContactMover():

    def __init__(self):
        self.joints_dict = {'rh_FFJ2': 0, 'rh_FFJ3': 0, 'rh_MFJ2': 1, 'rh_MFJ3': 1, 'rh_RFJ2': 2, 'rh_RFJ3': 2, 'rh_THJ5': 3}
        self.joints = list(self.joints_dict.keys())
        self.joint_publisher = rospy.Publisher('/hand/rh_trajectory_controller/command', JointTrajectory, queue_size = 10)
        self.biotac_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, callback = self.biotac_callback, queue_size = 10)
        self.joint_state_sub = rospy.Subscriber('/hand/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        self.current_biotac_values = [0, 0, 0, 0]
        self.current_joint_values = [0, 0, 0, 0, 0, 0, 0]
        self.current_joint_state = None
        self.initial_biotac_values = [0, 0, 0, 0]
        self.biotac_threshold = 5 # Value taken from biotac manual
        self.debug_pub = rospy.Publisher('/handover/debug/ccm', DisplayTrajectory, latch = True, queue_size = 1)
        #self.wait_for_initial_values()
        rospy.Service('handover/ccm', CCM, self.move_until_contacts)
        rospy.spin()

    def wait_for_initial_values(self):
        values_set = False
        while not values_set:
            values_set = 0 not in self.current_biotac_values 
        self.initial_biotac_values = self.current_biotac_values
        while self.current_joint_state is None:
            pass

    def biotac_callback(self, sensor_data):
        self.current_biotac_values[0] = sensor_data.tactiles[0].electrodes
        self.current_biotac_values[1] = sensor_data.tactiles[1].electrodes
        self.current_biotac_values[2] = sensor_data.tactiles[2].electrodes
        self.current_biotac_values[3] = sensor_data.tactiles[4].electrodes

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

    def move_until_contacts(self, req):
        contacts = [False, False, False, False]
        targets = [1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.0]
        steps = []
        for x in range(len(targets)):
            diff = targets[x] - self.current_joint_values[x]
            steps_temp = [self.current_joint_values[x] + y * diff/30 for y in range(30)]
            steps.append(steps_temp)

        # Publish trajectory for debugging
        debug_traj = DisplayTrajectory()
        debug_traj.trajectory = [RobotTrajectory()]
        debug_traj.trajectory[0].joint_trajectory = self.create_joint_trajectory_msg(self.joints, steps)
        debug_traj.trajectory_start.joint_state = rospy.wait_for_message('joint_states', JointState)
        self.debug_pub.publish(debug_traj)

        for x in range(len(steps[0])):
            used_joints = []
            used_steps = []
            for i in range(len(self.joints)):
                if not contacts[self.joints_dict[self.joints[i]]]:
                    used_joints.append(self.joints[i])
                    used_steps.append(steps[i][x])
            msg = self.create_joint_trajectory_msg(used_joints, used_steps)
            msg.header.stamp = rospy.Time.now()
            self.joint_publisher.publish(msg)
            # wait until movement finished -> convert to action client?
            for i in range(len(contacts)):
                if not contacts[i]:
                    if (self.current_biotac_values[i] > self.initial_biotac_values[i] + self.biotac_threshold) or (self.current_biotac_values[i] < self.initial_biotac_values[i] - self.biotac_threshold):
                        contacts[i] = True
            if sum(contacts) == 4:
                break
        return True

if __name__ == '__main__':
    rospy.init_node('close_contact_mover')
    ccm = CloseContactMover()
    ccm.move_until_contacts()
