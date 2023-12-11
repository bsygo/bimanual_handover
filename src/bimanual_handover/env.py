#!/usr/bin/env python3

import rospy
import random
from datetime import datetime
import rospkg
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Bool, Float32MultiArray, Float32, String
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import bimanual_handover.syn_grasp_gen as sgg
from bimanual_handover_msgs.srv import CCM, GraspTesterSrv, HandoverControllerSrv
import sensor_msgs.point_cloud2 as pc2
from moveit_msgs.msg import MoveItErrorCodes
from moveit_commander import MoveItCommanderException
from pr2_msgs.msg import PressureState
from sr_robot_msgs.msg import BiotacAll, MechanismStatistics
from sensor_msgs.msg import JointState
from copy import deepcopy
import os
import glob
import rosbag

# Environment used for training on the real robot
class RealEnv(gym.Env):

    def __init__(self, fingers, record = True, env_type = "tactile", time = None):
        super().__init__()
        rospy.on_shutdown(self.close)
        self.env_type = env_type

        # Setup action and observation space
        # 3 values for finger synergies
        self.action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype = np.float32)
        if self.env_type == "tactile":
            # First 5 values for biotac diff, last 3 values for current joint config in pca space
            self.observation_space = spaces.Box(low = -100, high = 100, shape = (8,), dtype = np.float32)
        elif self.env_type == "effort":
            # First 3 values for current joint config in pca space, next 9 values for efforts, last 3 values for one-hot encoding of object
            self.observation_space = spaces.Box(low = -1000, high = 1000, shape = (15,), dtype = np.float32)

        # Initialize observation values
        self.last_joints = None
        self.initial_tactile = None
        self.current_tactile = None
        self.initial_effort = None
        self.current_effort = None
        self.current_object = [0, 0, 0]

        # Setup information about relevant finger structure
        self.fingers = fingers
        self.closing_joints = ['rh_FFJ2', 'rh_FFJ3', 'rh_MFJ2', 'rh_MFJ3', 'rh_RFJ2', 'rh_RFJ3', 'rh_LFJ2', 'rh_LFJ3', 'rh_THJ2']
        self.joint_order = self.fingers.get_active_joints()

        # Setup observation callbacks
        self.tactile_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, self.tactile_callback)
        self.effort_sub = rospy.Subscriber('/hand/joint_states', JointState, self.effort_callback)
        #self.force_sub = rospy.Subscriber('/ft/l_gripper_motor', WrenchStamped, self.force_callback)
        #self.pressure_sub = rospy.Subscriber('/pressure/l_gripper_motor', PressureState, self.pressure_callback)

        # Setup relevant structures for action and reward generation
        self.pca_con = sgg.SynGraspGen()
        rospy.wait_for_service('grasp_tester_srv')
        self.gt_srv = rospy.ServiceProxy('grasp_tester_srv', GraspTesterSrv)

        # Setup interrupt option
        self.interrupt_sub = rospy.Subscriber('interrupt_learning', Bool, self.interrupt_callback)
        self.interrupted = False

        # Setup new object services
        rospy.wait_for_service('handover_controller_srv')
        self.handover_controller_srv = rospy.ServiceProxy('handover_controller_srv', HandoverControllerSrv)

        # Initialize reset parameters
        self.max_attempt_steps = 50
        self.current_attempt_step = 0
        self.reset_steps = 1000
        self.current_reset_step = 1000 # To set an object in first reset call

        # Further initializations if data should be recorded
        self.record = record
        if self.record:
            if time is None:
                self.time = datetime.now().strftime("%d_%m_%Y_%H_%M")
            else:
                self.time = time
            self.log_file = open(rospkg.RosPack().get_path('bimanual_handover') + "/logs/log" + self.time + ".txt", 'a')
            pkg_path = rospkg.RosPack().get_path('bimanual_handover')
            path = pkg_path + "/data/bags/"
            if os.path.isfile('{}training_{}.bag'.format(path, self.time)):
                self.bag = rosbag.Bag('{}training_{}.bag'.format(path, self.time), 'a')
            else:
                self.bag = rosbag.Bag('{}training_{}.bag'.format(path, self.time), 'w')


    def tactile_callback(self, tactile):
        self.current_tactile = [x.pdc for x in tactile.tactiles]

    def effort_callback(self, joint_state):
        self.current_effort = [joint_state.effort[joint_state.name.index(name)] for name in self.closing_joints]

    '''
    def force_callback(self, force):
        self.current_force = force
    '''

    '''
    def pressure_callback(self, pressure):
        self.current_pressure = pressure
    '''

    def interrupt_callback(self, command):
        self.interrupted = command.data

    def write_float_array_to_bag(self, topic, array):
        array_msg = Float32MultiArray()
        array_msg.data = array
        self.bag.write(topic, array_msg)

    def reset_hand_pose(self):
        if self.current_object[0] == 1:
            object_type = "can"
        elif self.current_object[1] == 1:
            object_type = "book"
        elif self.current_object[2] == 1:
            object_type = "unknown"
        self.handover_controller_srv("train", "pca", object_type)

    def step(self, action):
        while self.interrupted:
            rospy.sleep(1)

        # Check number of contacts
        if self.env_type == "tactile":
            current_tactile = deepcopy(self.current_tactile)
            current_tactile_diff = [current_tactile[x] - self.initial_tactile[x] for x in range(len(current_tactile))]
            contacts = [True if diff >=20 else False for diff in current_tactile_diff]
            contact_threshold = 5
        elif self.env_type == "effort":
            current_effort = deepcopy(self.current_effort)
            current_effort_diff = [abs((current_effort[x]) - (self.initial_effort[x])) for x in range(len(current_effort))]
            contacts = [True if diff >=150 else False for diff in current_effort_diff]
            contact_threshold = 9

        # Stop if enough contacts have been made or max steps were reached
        if sum(contacts) >= contact_threshold or self.current_attempt_step >= self.max_attempt_steps:
            terminated = True

            # Give reward depending on success of grasp
            success = self.gt_srv('y', True).success
            if success:
                reward = 1
            else:
                reward = -1

            # Add bonus reward depending on number of contacts
            if self.env_type == "tactile":
                reward += 0.1 * sum(contacts)
            elif self.env_type == "effort":
                reward += 0.1 * sum(contacts)/2

            # Set termination reason
            if self.current_attempt_step >= self.max_attempt_steps:
                terminated_reason = "Max steps"
            else:
                terminated_reason = "Contacts"
            print("Reward: {}, Terminated Reason: {}".format(reward, terminated_reason))
        # Continue if not enough contacts were made and not enough steps were done
        else:
            terminated = False

            # Reduce normalized actions
            action[0] = (action[0] - 1)/2
            action[1] = (action[1] - 1)/2

            # Turn action into joint configuration
            result = self.pca_con.gen_joint_config(action[:3], normalize = True)

            # Remove wrist joints
            del result['rh_WRJ1']
            del result['rh_WRJ2']

            # Remove fingers with contact
            if self.env_type == "tactile":
                checks = contacts
            elif self.env_type == "effort":
                checks = [contacts[0] and contacts[1], contacts[2] and contacts[3], contacts[4] and contacts[5], contacts[6] and contacts[7], contacts[8]]
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

            try:
                # Move into desired joint configuration
                self.fingers.set_joint_value_target(result)
                self.fingers.go()

                # Determine reward based on how much the fingers closed compared to the previous configuration
                reward = 0
                joint_diff = 0
                # Sometimes the error "Failed to fetch current robot state" occurs here
                # Possibly because finger joint values are not published in every joint_states msg.
                # This should ensure to try again until a correct state was received.
                current_joints = []
                while not current_joints:
                    current_joints = deepcopy(self.fingers.get_current_joint_values())
                for joint in self.closing_joints:
                    index = self.joint_order.index(joint)
                    if current_joints[index] >= self.last_joints[index]:
                        joint_diff += math.dist([current_joints[index]], [self.last_joints[index]])
                    else:
                        joint_diff += - math.dist([current_joints[index]], [self.last_joints[index]])
                reward = joint_diff
                print("Reward: {}".format(reward))
            # Terminate with negative reward if desired movement failed
            except MoveItCommanderException as e:
                rospy.logerr("Exception encountered: {}".format(e))
                terminated = True
                terminated_reason = "Exception {} encountered".format(e)
                reward = -1
            except IndexError as e:
                rospy.logerr("Error encountered: {}".format(e))
                rospy.logerr("current_joints list is the most likely error source. Value is {}.".format(current_joints))
                rospy.logerr("Next step is to terminate current run. Keeping it running until manual termination for diagnosis.")
                while not rospy.is_shutdown():
                    rospy.sleep(1)

        # Update observation for next step
        if self.env_type == "tactile":
            current_tactile = deepcopy(self.current_tactile)
            current_observation = [current_tactile[x] - self.initial_tactile[x] for x in range(len(current_tactile))]
        elif self.env_type == "effort":
            current_effort = deepcopy(self.current_effort)
            current_observation = [current_effort[x] - self.initial_effort[x] for x in range(len(current_effort))]
        current_pca = self.pca_con.get_pca_config()[0][:3]
        for pca in current_pca:
            current_observation.append(pca)
        for value in self.current_object:
            current_observation.append(value)
        observation = np.array(current_observation, dtype = np.float32)

        # Update step values
        self.current_attempt_step += 1
        self.current_reset_step += 1
        self.last_joints = deepcopy(self.fingers.get_current_joint_values())

        # Write data of this step if data recording is desired
        if self.record:
            # Write into log file
            if not terminated:
                terminated_reason = "False"
            self.log_file.write("Action: {}, Observation:  {}, Reward: {}, Terminated: {}, Object: {} \n".format(action, observation, reward, terminated_reason, self.current_object))

            # Write into rosbag
            self.write_float_array_to_bag('action', action)
            self.write_float_array_to_bag('obs', observation)
            reward_msg = Float32()
            reward_msg.data = reward
            self.bag.write('reward', reward_msg)
            terminated_reason_msg = String()
            terminated_reason_msg.data = terminated_reason
            self.bag.write('terminated', terminated_reason_msg)
            if self.env_type == "tactile":
                current_effort = deepcopy(self.current_effort)
                effort_diff = [current_effort[x] - self.initial_effort[x] for x in range(len(current_effort))]
                self.write_float_array_to_bag('effort', effort_diff)
            elif self.env_type == "effort":
                current_tactile = deepcopy(self.current_tactile)
                tactile_diff = [current_tactile[x] - self.initial_tactile[x] for x in range(len(current_tactile))]
                self.write_float_array_to_bag('tactile', tactile_diff)
            self.write_float_array_to_bag('object', self.current_object)

        info = {}
        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        while self.interrupted:
            rospy.sleep(1)

        # Reset hand into default position
        self.fingers.set_named_target('open')
        joint_values = dict(rh_THJ4 = 1.13446, rh_LFJ4 = -0.31699402670752413, rh_FFJ4 = -0.23131151280059523, rh_MFJ4 = 0.008929532157657268, rh_RFJ4 = -0.11378487918959583)
        self.fingers.set_joint_value_target(joint_values)
        self.fingers.go()

        # Set new initial values
        self.initial_tactile = deepcopy(self.current_tactile)
        self.initial_effort = deepcopy(self.current_effort)
        self.last_joints = deepcopy(self.fingers.get_current_joint_values())

        # Set new observation
        if self.env_type == "tactile":
            current_tactile = deepcopy(self.current_tactile)
            observation = [current_tactile[x] - self.initial_tactile[x] for x in range(len(current_tactile))]
        elif self.env_type == "effort":
            current_effort = deepcopy(self.current_effort)
            observation = [current_effort[x] - self.initial_effort[x] for x in range(len(current_effort))]
        current_pca = self.pca_con.get_pca_config()[0][:3]
        for pca in current_pca:
            observation.append(pca)

        # Setup with new object specified by user input
        self.current_attempt_step = 0
        if self.current_reset_step >= self.reset_steps:
            # Reset, insert new object, move to pose
            accepted_input = False
            while not accepted_input:
                # IDs: 1 - can, 2 - , 3 -
                object_id = input("Please enter the id of the next used object:")
                if object_id in ["1", "2", "3"]:
                    check = input("Object set to {}. Press enter to continue. If the wrong object was selected, please enter [a].".format(object_id))
                    if not check == "a":
                        # Set one-hot encoding
                        if object_id == "1":
                            one_hot = [1, 0, 0]
                        elif object_id == "2":
                            one_hot = [0, 1, 0]
                        elif object_id == "3":
                            one_hot = [0, 0, 1]
                        accepted_input = True
                    else:
                        print("Object selection aborted.")
                elif rospy.is_shutdown():
                    self.close()
                else:
                    print("Object id {} is not known. Please enter one of the known ids [1], [2] or [3].".format(object_id))
            self.current_object = one_hot
            for value in one_hot:
                observation.append(value)
            self.current_reset_step = 0
            self.reset_hand_pose()
        else:
            for value in self.current_object:
                observation.append(value)

        observation = np.array(observation, dtype = np.float32)
        info = {}
        return (observation, info)

    def render(self):
        return

    def close(self):
        if self.record:
            self.log_file.close()
            self.bag.close()
        return

# WIP Trajectory approach based environment, probably wont be used and more as a backup
class TrajEnv(gym.Env):

    def __init__(self, fingers):
        super().__init__()
        self.action_space = spaces.Box(low = -1, high = 1, shape = (6,), dtype = np.float32) # first 3 values for starting pose, last 3 for trajectory generation
        self.observation_space = None #Pointcloud
        self.fingers = fingers
        self.pca_con = sgg.SynGraspGen()
        self.time = datetime.now().strftime("%d_%m_%Y_%H_%M")
        self.log_file = open(rospkg.RosPack().get_path('bimanual_handover') + "/logs/log" + self.time + ".txt", 'w')
        self.initial_biotac = None
        self.joint_order = self.fingers.get_active_joints()
        self.tactile_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, self.tactile_callback)
        rospy.wait_for_service('grasp_tester')
        self.gt_srv = rospy.ServiceProxy('grasp_tester', GraspTesterSrv)
        self.interrupt_sub = rospy.Subscriber('handover/interrupt_learning', Bool, self.interrupt_callback)
        self.interrupted = False
        self.max_steps = 20
        self.current_step = 0

    def step(self, action):
        while self.interrupted:
            rospy.sleep(1)

        # Reduce normalized actions
        action[0] = (action[0] - 1)/2
        action[1] = (action[1] - 1)/2
        action[4] = (action[1] - 1)/2
        action[5] = (action[1] - 1)/2

        # Move into initial pose
        result = self.pca_con.gen_joint_config(action[:3], normalize = True)
        self.fingers.set_joint_value_target(result)
        self.fingers.go()

        # Stop moving if enough contacts have been made
        contacts = [False, False, False, False, False]
        while not sum(contacts >= 5):
            current_biotac = deepcopy(self.current_tactile)
            current_biotac_diff = [current_biotac[x] - self.initial_biotac[x] for x in range(len(current_biotac))]
            contacts = [True if diff >=20 else False for diff in current_biotac_diff]
            if sum(contacts) >= 5 or self.current_step >= self.max_steps:
                print(current_biotac_diff)
                success = self.gt_srv('y').success
                # Give reward if the previous decision was correct or not
                if success:
                    reward = 1
                else:
                    reward = -1
                if self.current_step >= self.max_steps:
                    terminated_reason = "Max steps"
                else:
                    terminated_reason = "Contacts"
                print("Reward: {}, Terminated Reason: {}".format(reward, terminated_reason))
            else:
                # Get new config
                result = self.pca_con.gen_joint_config(action[3:], normalize = True)
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

                # Move into desired config
                try:
                    self.fingers.set_joint_value_target(result)
                    self.fingers.go()
                except MoveItCommanderException as e:
                    rospy.logerr("Exception encountered: {}".format(e))
                    terminated_reason = "Exception {} encountered".format(e)
                    reward = -1
                    break

        # Update observation for next step
        observation = None #Pointcloud (irrelevant because of reset)

        self.current_step += 1

        self.log_file.write("Action: {}, Observation:  {}, Reward: {}, Terminated: {}".format(action, observation, reward, terminated_reason))

        info = {}
        truncated = False
        terminated = True
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.log_file.close()
        while self.interrupted:
            rospy.sleep(1)
        self.log_file = open(rospkg.RosPack().get_path('bimanual_handover') + "/logs/log"+ self.time + ".txt", 'a')

        # Reset hand into default position
        self.fingers.set_named_target('open')
        joint_values = dict(rh_THJ4 = 1.13446, rh_LFJ4 = -0.31699402670752413, rh_FFJ4 = -0.23131151280059523, rh_MFJ4 = 0.008929532157657268, rh_RFJ4 = -0.11378487918959583)
        self.fingers.set_joint_value_target(joint_values)
        self.fingers.go()

        self.initial_biotac = deepcopy(self.current_tactile)

        observation = None #Pointcloud

        self.current_step = 0

        info = {}
        return (observation, info)

    def render(self):
        return

    def close(self):
        self.log_file.close()
        return

# Env to be used to learn to imitate standard closing grasping, used for training in demo mode
class MimicEnv(gym.Env):

    def __init__(self, fingers):
        super().__init__()
        self.action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype = np.float32) # 3 values for finger synergies
        self.observation_space = spaces.Box(low = -10, high = 10, shape = (3,), dtype = np.float32) # Joint values encoded in pca space
        self.fingers = fingers
        self.pca_con = sgg.SynGraspGen()
        self.closing_joints = ['rh_FFJ2', 'rh_FFJ3', 'rh_MFJ2', 'rh_MFJ3', 'rh_RFJ2', 'rh_RFJ3', 'rh_LFJ2', 'rh_LFJ3', 'rh_THJ2']
        self.joint_order = self.fingers.get_active_joints()
        self.load_bags()
        self.current_index = None
        self.interrupt_sub = rospy.Subscriber('handover/interrupt_learning', Bool, self.interrupt_callback)
        self.interrupted = False

    def interrupt_callback(self, command):
        self.interrupted = command.data

    def load_bags(self):
        pkg_path = rospkg.RosPack().get_path('bimanual_handover')
        path = pkg_path + "/data/"
        obs_force_arr = []
        obs_joints_arr = []
        obs_pressure_arr = []
        obs_tactile_arr = []
        obs_tactile_init_arr = []
        finished_indices = []
        res_joints_arr = []
        for file_name in glob.iglob(f'{path}/closing_attempt_*.bag'):
            bag = rosbag.Bag(file_name)
            #init_tactile = None
            for topic, msg, t in bag.read_messages():
                if topic == 'obs_force':
                    obs_force_arr.append(msg)
                elif topic == 'obs_joints':
                    obs_joints_arr.append(msg)
                elif topic == 'obs_pressure':
                    obs_pressure_arr.append(msg)
                elif topic == 'obs_tactile':
                    #if init_tactile is None:
                    #    init_tactile = msg
                    obs_tactile_arr.append(msg)
                    #obs_tactile_init_arr.append(init_tactile)
                elif topic == 'res_joints':
                    res_joints_arr.append(msg)
            finished_indices.append(len(res_joints_arr))
            bag.close()
        finished_arr = [0] * len(res_joints_arr)
        for index in finished_indices:
            finished_arr[index-1] = 1
        self.obs = [obs_force_arr, obs_joints_arr, obs_pressure_arr, obs_tactile_arr, finished_arr]#, obs_tactile_init_arr]
        self.res = [res_joints_arr]
        return

    def step(self, action):
        while self.interrupted:
            rospy.sleep(1)
        result = self.pca_con.gen_joint_config(action)
        planned_joints = self.res[0][self.current_index]
        current_joints = [result[joint] for joint in planned_joints.name]
        observation = np.zeros((3,), dtype = np.float32)
        reward = 0
        for i in range(len(current_joints)):
            reward += -math.dist([current_joints[i]], [planned_joints.position[i]])

        info = {}
        terminated = True
        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        while self.interrupted:
            rospy.sleep(1)
        observation = {}
        # get data from random data entry
        self.current_index = random.randint(0, len(self.res[0])-1)
        joints = self.obs[1][self.current_index]
        pca_config = self.pca_con.get_pca_config(joints.position)[0][:3]
        observation = np.array(pca_config, dtype = np.float32)
        info = {}
        return (observation, info)

    def render(self):
        return

    def close(self):
        self.log_file.close()
        return

# DEPRECATED, UNFINISHED env for training to just learn initial pose from synergies and then close using ccm
class InitialEnv(gym.Env):

    def __init__(self, fingers):
        super().__init__()
        self.action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype = np.float32) # 3 values for finger synergies
        self.observation_space = spaces.Box(low = np.array(-100, -100, -100, -100, -100, -1, -1, -1),
                                            high = np.array(100, 100, 100, 100, 100, 1, 1, 1),
                                            dtype = np.flaot32) # First 5 values previous biotac diffs, last 3 values previous actions
        self.fingers = fingers
        self.pca_con = sgg.SynGraspGen()
        self.ccm_srv = rospy.ServiceProxy('handover/ccm', CCM)
        self.gt_srv = rospy.ServiceProxy('handover/grasp_tester', GraspTesterSrv)
        self.tactile_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, self.tactile_callback)

    def tactile_callback(self, tactile):
        self.current_tactile = [x.pdc for x in tactile.tactiles]

    def step(self, action):
        result = self.pca_con.gen_joint_config(action[:3])

        self.fingers.set_joint_value_target(result)
        self.fingers.go()

        self.ccm_srv('placeholder')
        success = self.gt_srv('y')

        if success:
            reward = 1
            terminated = True
        else:
            reward = -1
            terminated = False
            fingers.set_named_target('open')

        current_biotac = deepcopy(self.current_tactile)
        current_observation = [current_biotac[x] - self.initial_biotac[x] for x in range(len(current_biotac))]
        current_observation.append(x for x in action)
        observation = np.array(current_observation, dtype = np.float32)

        info = {}
        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        observation = np.zeros((8,), dtype.float32)
        self.initial_biotac = deepcopy(self.current_tactile)
        info = {}
        return (observation, info)

    def render(self):
        return

    def close(self):
        return

