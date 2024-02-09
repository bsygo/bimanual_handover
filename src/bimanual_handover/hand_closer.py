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
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sr_robot_msgs.msg import BiotacAll
from bimanual_handover_msgs.srv import HandCloserSrv
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from copy import deepcopy
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from sr_robot_msgs.msg import BiotacAll
from pr2_msgs.msg import PressureState
from geometry_msgs.msg import WrenchStamped

class DemoCloser():

    def __init__(self):
        # Setup fingers
        self.fingers = MoveGroupCommander('right_fingers', ns="/")
        self.fingers.set_max_velocity_scaling_factor(1.0)
        self.fingers.set_max_acceleration_scaling_factor(1.0)

        rospy.Service('hand_closer_srv', HandCloserSrv, self.move_fixed)
        rospy.spin()

    def move_fixed(self, req):
        finger_dict = {}
        finger_dict["rh_FFJ2"] = 0.785398
        finger_dict["rh_FFJ3"] = 0.785398
        finger_dict["rh_MFJ2"] = 0.785398
        finger_dict["rh_MFJ3"] = 0.785398
        finger_dict["rh_RFJ2"] = 0.785398
        finger_dict["rh_RFJ3"] = 0.785398
        finger_dict["rh_LFJ2"] = 0.785398
        finger_dict["rh_LFJ3"] = 0.785398
        finger_dict["rh_THJ5"] = 0.436332
        self.fingers.set_joint_value_target(finger_dict)
        self.fingers.go()
        return True

class PCACloser():

    def __init__(self):
        self.joint_client = actionlib.SimpleActionClient('/hand/rh_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.joint_client.wait_for_server()
        self.closing_joints = ['rh_FFJ2', 'rh_FFJ3', 'rh_MFJ2', 'rh_MFJ3', 'rh_RFJ2', 'rh_RFJ3', 'rh_LFJ2', 'rh_LFJ3', 'rh_THJ2']

        # Initialized based on desired mode
        self.mode = rospy.get_param("hand_closer/mode")
        if self.mode == "tactile":
            self.tactile_threshold = 20
            self.initial_tactile_values = None
            self.current_tactile_values = None
            self.tactile_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, self.tactile_callback)
        elif self.mode == "effort":
            self.effort_thresholds = [200, 150, 150, 150, 150, 150, 150, 150, 150]
            self.initial_effort_values = None
            self.current_effort_values = None
            self.effort_sub = rospy.Subscriber('/hand/joint_states', JointState, self.effort_callback)

        # Setup grasp generator
        self.sgg = sgg.SynGraspGen()
        self.constant_alphas = [-1.0, -0.5, 0.3]
        self.contacts_made = False

        # Start service
        rospy.Service('hand_closer_srv', HandCloserSrv, self.move_until_contacts)
        rospy.spin()

    def tactile_callback(self, tactile):
        self.current_tactile_values = [x.pdc for x in tactile.tactiles]

    def effort_callback(self, joint_state):
        self.current_effort_values = [joint_state.effort[joint_state.name.index(name)] for name in self.closing_joints]

    def wait_for_initial_values(self):
        if self.mode == "tactile":
            while self.current_tactile_values is None:
                pass
            self.initial_tactile_values = deepcopy(self.current_tactile_values)
        elif self.mode == "effort":
            while self.current_effort_values is None:
                pass
            self.initial_effort_values = deepcopy(self.current_effort_values)

    def move_step(self):
        # Stop moving if enough contacts have been made
        if self.mode == "tactile":
            current_tactile = deepcopy(self.current_tactile_values)
            current_tactile_diff = [abs((current_tactile[x]) - (self.initial_tactile_values[x])) for x in range(len(current_tactile))]
            contacts = [True if diff >=20 else False for diff in current_tactile_diff]
            contacts = []
            thresholds = [20, 20, 20, 20, 20]
            for i in range(len(current_tactile_diff)):
                contacts.append(current_tactile_diff[i] >= thresholds[i])
            contact_threshold = 5
        elif self.mode == "effort":
            current_effort = deepcopy(self.current_effort_values)
            current_effort_diff = [abs((current_effort[x]) - (self.initial_effort_values[x])) for x in range(len(current_effort))]
            contacts = []
            thresholds = [150, 150, 150, 150, 150, 150, 150, 150, 150]
            for i in range(len(current_effort_diff)):
                contacts.append(current_effort_diff[i] >= thresholds[i])
            contact_threshold = 9

        # Get new config
        result = self.sgg.gen_joint_config(self.constant_alphas)

        # Remove wrist joints
        del result['rh_WRJ1']
        del result['rh_WRJ2']

        # Remove fingers with contact
        if self.mode == "tactile":
            checks = contacts
        elif self.mode == "effort":
            checks = [contacts[0] and contacts[1], contacts[2] and contacts[3], contacts[4] and contacts[5], contacts[6] and contacts[7], contacts[8]]
        # Break early if all contacts have already been made
        if all(checks):
            self.contacts_made = True
            return
        del_keys = []
        if checks[0]:
            for key in result.keys():
                if 'rh_FFJ' in key:
                    del_keys.append(key)
        if checks[1]:
            for key in result.keys():
                if 'rh_MFJ' in key:
                    del_keys.append(key)
        if checks[2]:
            for key in result.keys():
                if 'rh_RFJ' in key:
                    del_keys.append(key)
        if checks[3]:
            for key in result.keys():
                if 'rh_LFJ' in key:
                    del_keys.append(key)
        if checks[4]:
            for key in result.keys():
                if 'rh_THJ' in key:
                    del_keys.append(key)
        for key in del_keys:
            del result[key]

        # Create Trjaectory msg
        point = JointTrajectoryPoint(positions = list(result.values()), time_from_start = rospy.Duration.from_sec(0.5))
        traj = JointTrajectory(joint_names = list(result.keys()), points = [point])
        follow_traj = FollowJointTrajectoryGoal(trajectory = traj)

        # Move to the trajectory
        self.joint_client.send_goal(follow_traj)
        self.joint_client.wait_for_result()

    def move_until_contacts(self, req):
        self.wait_for_initial_values()
        while not self.contacts_made:
            self.move_step()
        self.contacts_made = False
        return True

class ThresholdCloser():

    def __init__(self):
        # Setup joint control
        self.joints_dict = {'rh_FFJ2': 0, 'rh_FFJ3': 0, 'rh_MFJ2': 1, 'rh_MFJ3': 1, 'rh_RFJ2': 2, 'rh_RFJ3': 2, 'rh_LFJ2': 3, 'rh_LFJ3': 3, 'rh_THJ5': 4}
        self.closing_joints = list(self.joints_dict.keys())
        self.joint_client = actionlib.SimpleActionClient('/hand/rh_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.joint_client.wait_for_server()

        # Initialized based on desired mode
        self.mode = rospy.get_param("hand_closer/mode")
        if self.mode == "tactile":
            self.tactile_threshold = 20
            self.initial_tactile_values = None
            self.current_tactile_values = None
            self.tactile_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, self.tactile_callback)
        elif self.mode == "effort":
            self.effort_thresholds = [200, 150, 150, 150, 150, 150, 150, 150, 150]
            self.initial_effort_values = None
            self.current_effort_values = None
            self.effort_sub = rospy.Subscriber('/hand/joint_states', JointState, self.effort_callback)

        # Initialize joint state feedback
        self.current_joint_values = None
        self.current_joint_state = None
        self.joint_state_sub = rospy.Subscriber('/hand/joint_states', JointState, self.joint_callback)

        # Initialize debug and data collection
        self.debug = rospy.get_param("hand_closer/debug")
        self.collect = rospy.get_param("hand_closer/collect")
        if self.debug:
            self.debug_pub = rospy.Publisher('debug/hand_closer/trajectory', DisplayTrajectory, latch = True, queue_size = 1)
            self.debug_snapshot_pub = rospy.Publisher('debug/hand_closer/snapshot', Bool, queue_size = 1, latch = True)
            self.debug_snapshot_pub.publish(False)
        if self.collect:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('bimanual_handover')
            now = datetime.now()
            current_date = now.strftime("%d-%m-%Y-%H-%M-%S")
            name = "closing_attempt_" + current_date + ".bag"
            self.data_bag = Bag(pkg_path + "/data/" + name, 'w')

        # Start service
        rospy.Service('hand_closer_srv', HandCloserSrv, self.move_until_contacts)
        rospy.spin()

    def wait_for_initial_values(self):
        if self.debug:
            self.debug_snapshot_pub.publish(True)
        if self.mode == "tactile":
            while self.current_tactile_values is None:
                pass
            self.initial_tactile_values = deepcopy(self.current_tactile_values)
        elif self.mode == "effort":
            while self.current_effort_values is None:
                pass
            self.initial_effort_values = deepcopy(self.current_effort_values)
        while self.current_joint_state is None:
            pass

    def tactile_callback(self, tactile):
        self.current_tactile_values = [x.pdc for x in tactile.tactiles]

    def effort_callback(self, joint_state):
        self.current_effort_values = [joint_state.effort[joint_state.name.index(name)] for name in self.closing_joints]

    def joint_callback(self, joint_state):
        indices = [joint_state.name.index(joint_name) for joint_name in self.closing_joints]
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
        # Initialize values
        finger_contacts = [False, False, False, False, False]
        joint_contacts = [False, False, False, False, False, False, False, False, False]
        targets = [1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.0]

        # Move joints a small step for deadband compensation
        msg = self.create_joint_trajectory_goal_msg(self.closing_joints, [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        self.joint_client.send_goal(msg)
        self.joint_client.wait_for_result()

        # Take snapshot of initial values after deadband compensation
        self.wait_for_initial_values()

        # Generate trajectory steps for each joint
        steps = []
        for x in range(len(targets)):
            diff = targets[x] - self.current_joint_values[x]
            steps_temp = [self.current_joint_values[x] + i * diff/30 for i in range(30)]
            steps.append(steps_temp)

        # Publish generated trajectory for debugging
        if self.debug:
            # Publish trajectory for debugging
            debug_traj = DisplayTrajectory()
            debug_traj.trajectory = [RobotTrajectory()]
            debug_traj.trajectory[0].joint_trajectory = self.create_joint_trajectory_msg(self.closing_joints, steps)
            debug_traj.trajectory_start.joint_state = self.wait_for_hand_joints()
            self.debug_pub.publish(debug_traj)

        # Go through each generated trajectory step
        for x in range(len(steps[0])):
            # Write data into rosbag for later reuse
            if self.collect:
                pressure = rospy.wait_for_message('/pressure/l_gripper_motor', PressureState)
                tactile = rospy.wait_for_message('/hand/rh/tactile', BiotacAll)
                force = rospy.wait_for_message('/ft/l_gripper_motor', WrenchStamped)
                joints = self.wait_for_hand_joints()
                self.data_bag.write('obs_pressure', pressure)
                self.data_bag.write('obs_tactile', tactile)
                self.data_bag.write('obs_force', force)
                self.data_bag.write('obs_joints', joints)

            # Decide which joints still need to be moved
            used_joints = []
            used_steps = []
            for i in range(len(self.closing_joints)):
                if not finger_contacts[self.joints_dict[self.closing_joints[i]]]:
                    used_joints.append(self.closing_joints[i])
                    used_steps.append(steps[i][x])

            # Move joints one step further
            msg = self.create_joint_trajectory_goal_msg(used_joints, used_steps)
            self.joint_client.send_goal(msg)
            self.joint_client.wait_for_result()

            # Test for each joint if their finger has made contact
            for i in range(len(self.closing_joints)):
                if not joint_contacts[i]:
                    if self.mode == "tactile":
                        if abs(self.current_tactile_values[i] - self.initial_tactile_values[i]) > self.tactile_threshold:
                            joint_contacts[i] = True
                            rospy.loginfo('Contact found with joint {}.'.format(i))
                    elif self.mode == "effort":
                        if abs(self.current_effort_values[i] - self.initial_effort_values[i]) > self.effort_thresholds[i]:
                            joint_contacts[i] = True
                            rospy.loginfo('Contact found with joint {}.'.format(i))

            # Update finger contacts
            finger_contacts = [joint_contacts[0] and joint_contacts[1], joint_contacts[2] and joint_contacts[3], joint_contacts[4] and joint_contacts[5], joint_contacts[6] and joint_contacts[7], joint_contacts[8]]

            # Write joint values after movement into the rosbag
            if self.collect:
                joints = self.wait_for_hand_joints()
                self.data_bag.write('res_joints', joints)

            # Stop if all fingers have made contact
            if sum(finger_contacts) == 5:
                rospy.loginfo('contacts reached')
                # Move joints a small step to apply more pressure to the object
                goal_values = self.current_joint_values
                goal_values = [value + 0.05 for value in goal_values]
                msg = self.create_joint_trajectory_goal_msg(self.closing_joints, goal_values)
                self.joint_client.send_goal(msg)
                self.joint_client.wait_for_result()

                if self.collect:
                    self.data_bag.close()
                return True

        # Return failure if any finger still hsan't made contact
        if self.collect:
            self.data_bag.close()
        return True

class ModelCloser():

    def __init__(self):
        # Setup and load model paths
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('bimanual_handover')
        model_path = pkg_path + rospy.get_param("hand_closer/model_closer/model_path")
        self.model = SAC.load(model_path)
        self.model.load_replay_buffer(pkg_path + rospy.get_param("hand_closer/model_closer/model_replay_buffer_path"))
        self.contact_modality = rospy.get_param("hand_closer/model_closer/contact_modality")

        # Set observation inputs
        self.joint_states_input = rospy.get_param("hand_closer/model_closer/observation_space/joint_states")
        self.pca_input = rospy.get_param("hand_closer/model_closer/observation_space/pca")
        self.effort_input = rospy.get_param("hand_closer/model_closer/observation_space/effort")
        self.tactile_input = rospy.get_param("hand_closer/model_closer/observation_space/tactile")
        self.one_hot_input = rospy.get_param("hand_closer/model_closer/observation_space/one_hot")

        # Setup fingers
        self.fingers = MoveGroupCommander('right_fingers', ns="/")
        self.fingers.set_max_velocity_scaling_factor(1.0)
        self.fingers.set_max_acceleration_scaling_factor(1.0)

        # Set initial values
        self.current_object = [0, 0, 0]
        self.initial_effort = None
        self.current_effort = None
        self.initial_tactile = None
        self.current_tactile = None
        self.current_pca = None
        self.current_joint_values = None
        self.contacts_made = False

        # Setup grasp generator
        self.pca_con = sgg.SynGraspGen()

        # Settup relevant subscribers
        self.closing_joints = ['rh_FFJ2', 'rh_FFJ3', 'rh_MFJ2', 'rh_MFJ3', 'rh_RFJ2', 'rh_RFJ3', 'rh_LFJ2', 'rh_LFJ3', 'rh_THJ2']
        self.tactile_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, self.tactile_callback)
        self.effort_sub = rospy.Subscriber('/hand/joint_states', JointState, self.effort_callback)
        self.joint_values_sub = rospy.Subscriber('/hand/joint_states', JointState, callback = self.joint_values_callback)

        # Start service
        rospy.loginfo("hand_closer_srv ready.")
        rospy.Service('hand_closer_srv', HandCloserSrv, self.close_hand)
        rospy.spin()

    def tactile_callback(self, tactile):
        self.current_tactile = [x.pdc for x in tactile.tactiles]

    def effort_callback(self, joint_state):
        self.current_effort = [joint_state.effort[joint_state.name.index(name)] for name in self.closing_joints]

    def joint_values_callback(self, joint_state):
        self.current_joint_values = [joint_state.position[joint_state.name.index(name)] for name in self.closing_joints]

    def get_next_action(self):
        # Update observation for next step
        observation = []

        # Add joint values
        if self.joint_values_input:
            for value in self.current_joint_values:
                observation.append(value)

        # Add pca values
        if self.pca_input:
            current_pca = self.pca_con.get_pca_config()[0][:3]
            for pca in current_pca:
                observation.append(pca)

        # Add tactile values
        if self.tactile_input:
            current_tactile = deepcopy(self.current_tactile)
            tactile_diff = [current_tactile[x] - self.initial_tactile[x] for x in range(len(current_tactile))]
            # Normalize diff values between -1 and 1
            for diff in tactile_diff:
                if diff > 40:
                    observation.append(1)
                elif diff < -40:
                    observation.append(-1)
                else:
                    observation.append(diff/40)

        # Add effort values
        if self.effort_input:
            current_effort = deepcopy(self.current_effort)
            effort_diff = [current_effort[x] - self.initial_effort[x] for x in range(len(current_effort))]
            # Normalize diff values between -1 and 1
            for diff in effort_diff:
                if diff > 300:
                    observation.append(1)
                elif diff < -300:
                    observation.append(-1)
                else:
                    observation.append(diff/300)

        # Add one-hot encoding
        if self.one_hot_input:
            for value in self.current_object:
                current_observation.append(value)
        observation = np.array(current_observation, dtype = np.float32)
        action, _states = self.model.predict(observation, deterministic = True)
        return action

    def exec_action(self, action):
        # Stop moving if enough contacts have been made
        if self.contact_modality == "tactile":
            current_tactile = deepcopy(self.current_tactile)
            current_tactile_diff = [abs((current_tactile[x]) - (self.initial_tactile[x])) for x in range(len(current_tactile))]
            contacts = [True if diff >=20 else False for diff in current_tactile_diff]
            contacts = []
            thresholds = [20, 20, 20, 20, 20]
            for i in range(len(current_tactile_diff)):
                contacts.append(current_tactile_diff[i] >= thresholds[i])
            contact_threshold = 5
        elif self.contact_modality == "effort":
            current_effort = deepcopy(self.current_effort)
            current_effort_diff = [abs((current_effort[x]) - (self.initial_effort[x])) for x in range(len(current_effort))]
            contacts = []
            thresholds = [150, 150, 150, 150, 150, 150, 150, 150, 150]
            for i in range(len(current_effort_diff)):
                contacts.append(current_effort_diff[i] >= thresholds[i])
            contact_threshold = 9

        # Reduce normalized actions
        action[0] = (action[0] - 1)/2
        action[1] = (action[1] - 1)/2

        # Get new config
        result = self.pca_con.gen_joint_config(action[:3], normalize = True)

        # Remove wrist joints
        del result['rh_WRJ1']
        del result['rh_WRJ2']

        # Remove fingers with contact
        if self.contact_modality == "tactile":
            checks = contacts
        elif self.contact_modality == "effort":
            checks = [contacts[0] and contacts[1], contacts[2] and contacts[3], contacts[4] and contacts[5], contacts[6] and contacts[7], contacts[8]]
        # Break early if all contacts have already been made
        if all(checks):
            self.contacts_made = True
            return
        del_keys = []
        if checks[0]:
            for key in result.keys():
                if 'rh_FFJ' in key:
                    del_keys.append(key)
        if checks[1]:
            for key in result.keys():
                if 'rh_MFJ' in key:
                    del_keys.append(key)
        if checks[2]:
            for key in result.keys():
                if 'rh_RFJ' in key:
                    del_keys.append(key)
        if checks[3]:
            for key in result.keys():
                if 'rh_LFJ' in key:
                    del_keys.append(key)
        if checks[4]:
            for key in result.keys():
                if 'rh_THJ' in key:
                    del_keys.append(key)
        for key in del_keys:
            del result[key]

        # Move fingers to configuration
        self.fingers.set_joint_value_target(result)
        self.fingers.go()

    def close_hand(self, req):
        # Set object to the requested one
        if req.object_type == "can":
            self.current_object = [1, 0, 0]
        elif req.object_type == "book":
            self.current_object = [0, 1, 0]

        # Set initial values
        self.initial_tactile = deepcopy(self.current_tactile)
        self.initial_effort = deepcopy(self.current_effort)

        # Generate and execute actions
        while not self.contacts_made:
            action = self.get_next_action()
            self.exec_action(action)

        self.contacts_made = False
        return True

def shutdown():
    roscpp_shutdown()

def init_mover():
    rospy.init_node('hand_closer')
    roscpp_initialize('')
    rospy.on_shutdown(shutdown)

    mode = rospy.get_param("hand_closer/closer_type")
    if mode == "threshold":
        ThresholdCloser()
    elif mode == "model":
        ModelCloser()
    elif mode == "demo":
        DemoCloser()
    elif mode == "pca":
        PCACloser()
    else:
        rospy.logerr("Unknown closer type: {}".format(mode))

if __name__ == '__main__':
    init_mover()
