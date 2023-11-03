#!/usr/bin/env python3

import rospy
import rospkg
import math
import sys
import numpy as np
from stable_baselines3 import PPO, SAC
import bimanual_handover.syn_grasp_gen as sgg
from rosbag import Bag
from datetime import datetime
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_msgs.msg import BiotacAll
from bimanual_handover_msgs.srv import HandCloserSrv
from moveit_commander import MoveGroupCommander
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
            current_date = now.strftime("%d-%m-%Y-%H-%M-%S")
            name = "closing_attempt_" + current_date + ".bag"
            self.data_bag = Bag(pkg_path + "/data/" + name, 'w')
        self.joint_client.wait_for_server()
        rospy.Service('hand_closer_src', HandCloserSrv, self.move_until_contacts)
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
        modified_data = sensor_data.tactiles[3].electrodes[:6] + sensor_data.tactiles[3].electrodes[10:13] + sensor_data.tactiles[3].electrodes[16:17] # volatile biotac sensors
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

    def wait_for_hand_joints(self):
        received = False
        while not received:
            joint_states = rospy.wait_for_message('/joint_states', JointState)
            if 'rh_FFJ2' in joint_states.name:
                received = True
        return joint_states

    def move_until_contacts(self, req):
        rospy.sleep(2)
        self.wait_for_initial_values()
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
            debug_traj.trajectory_start.joint_state = self.wait_for_hand_joints()
            self.debug_pub.publish(debug_traj)

        for x in range(len(steps[0])):
            if self.collect:
                pressure = rospy.wait_for_message('/pressure/l_gripper_motor', PressureState)
                tactile = rospy.wait_for_message('/hand/rh/tactile', BiotacAll)
                force = rospy.wait_for_message('/ft/l_gripper_motor', WrenchStamped)
                joints = self.wait_for_hand_joints()
                self.data_bag.write('obs_pressure', pressure)
                self.data_bag.write('obs_tactile', tactile)
                self.data_bag.write('obs_force', force)
                self.data_bag.write('obs_joints', joints)
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
                joints = self.wait_for_hand_joints()
                self.data_bag.write('res_joints', joints)
            if sum(contacts) == 5:
                print('contacts reached')
                if self.collect:
                    self.data_bag.close()
                return True
        if self.collect:
            self.data_bag.close()
        print(contacts)
        return False

class ModelMover():

    def __init__(self, model_name = "sac_checkpoint_23_10_2023_15_47_12900_steps.zip"):
        rospy.init_node('model_mover')
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('bimanual_handover')
        model_path = pkg_path + "/models/" + "checkpoints/" + model_name
        self.model = SAC.load(model_path)
        self.model.load_replay_buffer(pkg_path + "/models/checkpoints/sac_checkpoint_23_10_2023_15_47_replay_buffer_12900_steps")
        self.fingers = MoveGroupCommander('right_fingers', ns="/")
        self.fingers.set_max_velocity_scaling_factor(1.0)
        self.fingers.set_max_acceleration_scaling_factor(1.0)
        self.initial_tactile = None
        self.current_tactile = None
        self.current_pca = None
        self.tactile_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, self.tactile_callback)
        self.joints = ['rh_WRJ2', 'rh_WRJ1', 'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1', 'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1', 'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1', 'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1', 'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1']
        self.pca_con = sgg.SynGraspGen()
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        #self.joint_state_sub = rospy.Subscriber('/hand/joint_states', JointState, callback = self.joint_callback, queue_size = 10)
        rospy.Service('hand_closer_srv', HandCloserSrv, self.close_hand)
        rospy.spin()

    def tactile_callback(self, tactile):
        self.current_tactile = [x.pdc for x in tactile.tactiles]

    def joint_callback(self, joint_state):
        if not self.joints[0] in joint_state.name:
            return
        indices = [joint_state.name.index(joint_name) for joint_name in self.joints]
        current_joint_values = [joint_state.position[x] for x in indices]
        current_joint_values = np.array(current_joint_values, dtype = np.float32)
        self.current_pca = self.pca_con.get_pca_config(current_joint_values)[0][:3]

    def get_next_action(self):
        current_biotac = deepcopy(self.current_tactile)
        current_observation = [current_biotac[x] - self.initial_tactile[x] for x in range(len(current_biotac))]
        current_pca = self.pca_con.get_pca_config()[0][:3]
        for pca in current_pca:
            current_observation.append(pca)
        observation = np.array(current_observation, dtype = np.float32)
        action, _states = self.model.predict(observation, deterministic = True)
        return action

    def exec_action(self, action):
        # Stop moving if enough contacts have been made
        current_biotac = deepcopy(self.current_tactile)
        current_biotac_diff = [current_biotac[x] - self.initial_tactile[x] for x in range(len(current_biotac))]
        contacts = [True if diff >=20 else False for diff in current_biotac_diff]
        # Reduce normalized actions
        action[0] = (action[0] - 1)/2
        action[1] = (action[1] - 1)/2
        # Get new config
        result = self.pca_con.gen_joint_config(action[:3], normalize = True)
        # Remove wrist joints
        del result['rh_WRJ1']
        del result['rh_WRJ2']

        # Remove fingers with contact (optional, testing) -> Mabe switch to not close anymore instead of no movement at all
        del_keys = []
        if contacts[0]:
            for key in result.keys():
                if 'rh_FFJ' in key:
                    del_keys.append(key)
        if contacts[1]:
            for key in result.keys():
                if 'rh_MFJ' in key:
                    del_keys.append(key)
        if contacts[2]:
            for key in result.keys():
                if 'rh_RFJ' in key:
                    del_keys.append(key)
        if contacts[3]:
            for key in result.keys():
                if 'rh_LFJ' in key:
                    del_keys.append(key)
        if contacts[4]:
            for key in result.keys():
                if 'rh_THJ' in key:
                    del_keys.append(key)
        for key in del_keys:
            del result[key]

        self.fingers.set_joint_value_target(result)
        self.fingers.go()

    def close_hand(self, req):
        self.initial_tactile = deepcopy(self.current_tactile)
        for i in range(20):
            action = self.get_next_action()
            self.exec_action(action)
        return True

def init_mover(mode):
    if mode == "ccm":
        CloseContactMover()
    elif mode == "pca":
        ModelMover()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        rospy.logerr("Missing parameters to call hand_closer. Exiting.")
    else:
        init_mover(sys.argv[1]) 
