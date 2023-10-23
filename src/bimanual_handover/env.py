#!/usr/bin/env python3

import rospy
import random
from datetime import datetime
import rospkg
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Bool
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import bimanual_handover.syn_grasp_gen as sgg
from bimanual_handover_msgs.srv import CCM, GraspTesterSrv
import sensor_msgs.point_cloud2 as pc2
from moveit_msgs.msg import MoveItErrorCodes
from moveit_commander import MoveItCommanderException
from pr2_msgs.msg import PressureState
from sr_robot_msgs.msg import BiotacAll
from copy import deepcopy
import os
import glob
import rosbag

class RealEnv(gym.Env):

    def __init__(self, fingers):
        super().__init__()
        self.action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype = np.float32) # first 3 values for finger synergies, last value if grasp is final
        self.observation_space = spaces.Box(low = -100, high = 100, shape = (8,), dtype = np.float32) # First 5 values for biotac diff, last 3 values for current joint config in pca space
        self.fingers = fingers
        self.pca_con = sgg.SynGraspGen()
        self.time = datetime.now().strftime("%d_%m_%Y_%H_%M")
        self.log_file = open(rospkg.RosPack().get_path('bimanual_handover') + "/logs/log" + self.time + ".txt", 'w')
        self.last_joints = None
        self.initial_biotac = None
        self.closing_joints = ['rh_FFJ2', 'rh_FFJ3', 'rh_MFJ2', 'rh_MFJ3', 'rh_RFJ2', 'rh_RFJ3', 'rh_LFJ2', 'rh_LFJ3', 'rh_THJ2']
        self.joint_order = self.fingers.get_active_joints()
        self.ccm_srv = rospy.ServiceProxy('ccm', CCM)
        #self.pressure_sub = rospy.Subscriber('/pressure/l_gripper_motor', PressureState, self.pressure_callback)
        self.tactile_sub = rospy.Subscriber('/hand/rh/tactile', BiotacAll, self.tactile_callback)
        #self.force_sub = rospy.Subscriber('/ft/l_gripper_motor', WrenchStamped, self.force_callback)
        rospy.wait_for_service('grasp_tester')
        self.gt_srv = rospy.ServiceProxy('grasp_tester', GraspTesterSrv)
        self.interrupt_sub = rospy.Subscriber('handover/interrupt_learning', Bool, self.interrupt_callback)
        self.interrupted = False
        self.max_steps = 20
        self.current_step = 0

    '''
    def pressure_callback(self, pressure):
        self.current_pressure = pressure
    '''

    def tactile_callback(self, tactile):
        self.current_tactile = [x.pdc for x in tactile.tactiles]

    '''
    def force_callback(self, force):
        self.current_force = force
    '''

    def interrupt_callback(self, command):
        self.interrupted = command.data

    def step(self, action):
        while self.interrupted:
            rospy.sleep(1)

        # Check if networked determined to have a finished grasped
        #if action[3] >= 0.0:

        # Stop moving if enough contacts have been made
        current_biotac = deepcopy(self.current_tactile)
        current_biotac_diff = [current_biotac[x] - self.initial_biotac[x] for x in range(len(current_biotac))]
        contacts = [True if diff >=20 else False for diff in current_biotac_diff]
        if sum(contacts) >= 5 or self.current_step >= self.max_steps:
            print(current_biotac_diff)
            success = self.gt_srv('placeholder').success
            terminated = True
            # Give reward if the previous decision was correct or not
            if success:
                reward = 1
            else:
                reward = -1
            reward += 0.1 * sum(contacts)
            if self.current_step >= self.max_steps:
                terminated_reason = "Max steps"
            else:
                terminated_reason = "Contacts"
            print("Reward: {}, Terminated Reason: {}".format(reward, terminated_reason))
        else:
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
            
            # Move into desired config
            try:
                self.fingers.set_joint_value_target(result)
                self.fingers.go()

                # Determine reward based on how much the fingers closed compared to the previous configuration
                reward = 0
                terminated = False
                joint_diff = 0
                current_joints = self.fingers.get_current_joint_values()
                for joint in self.closing_joints:
                    index = self.joint_order.index(joint)
                    if current_joints[index] >= self.last_joints[index]:
                        joint_diff += math.dist([current_joints[index]], [self.last_joints[index]])
                    else:
                        joint_diff += - math.dist([current_joints[index]], [self.last_joints[index]])
                reward = joint_diff
                print("Reward: {}".format(reward))
            except MoveItCommanderException as e:
                rospy.logerr("Exception encountered: {}".format(e))
                terminated = True
                terminated_reason = "Exception {} encountered".format(e)
                reward = -1


        # Update observation for next step
        current_biotac = deepcopy(self.current_tactile)
        current_observation = [current_biotac[x] - self.initial_biotac[x] for x in range(len(current_biotac))]
        current_pca = self.pca_con.get_pca_config()[0][:3]
        for pca in current_pca:
            current_observation.append(pca)
        observation = np.array(current_observation, dtype = np.float32)

        self.current_step += 1
        self.last_joints = deepcopy(self.fingers.get_current_joint_values())

        if not terminated:
            terminated_reason = "False"
        self.log_file.write("Action: {}, Observation:  {}, Reward: {}, Terminated: {}".format(action, observation, reward, terminated_reason))

        info = {}
        truncated = False
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
        self.last_joints = deepcopy(self.fingers.get_current_joint_values())

        current_biotac = deepcopy(self.current_tactile)
        current_observation = [current_biotac[x] - self.initial_biotac[x] for x in range(len(current_biotac))]
        current_pca = self.pca_con.get_pca_config()[0][:3]
        for pca in current_pca:
            current_observation.append(pca)
        observation = np.array(current_observation, dtype = np.float32)

        self.current_step = 0

        info = {}
        return (observation, info)

    def render(self):
        return

    def close(self):
        self.log_file.close()
        return

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
                success = self.gt_srv('placeholder').success
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
        success = self.gt_srv('placeholder')

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

class SimpleEnv(gym.Env):

    def __init__(self, fingers, pc, ps = None):
        super().__init__()
        self.action_space = spaces.Box(low = -1, high = 1, shape = (5,), dtype = np.float64) # + decision if finished, limits found through manual testing
        self.observation_space = spaces.Dict({"finger_contacts": spaces.MultiBinary(5)})
        # Should instead include force/torque readings from the gripper, the gripper pressure sensor data and the biotac tactile data

        #"finger_joints": spaces.Box(low = -1, high = 1, shape = (22,), dtype = np.float64), spaces.Box(-1.134, 1.658, shape = (22,), dtype = np.float64), "finger_contacts": spaces.MultiBinary(5)})# (last_action, finger_joints, finger_contacts) {"last_action": spaces.Box(low = np.array([-1.5, -0.65, -1.25]), high = np.array([1.5, 0.65, 1.25]), dtype = np.float64),
        self.fingers = fingers
        self.pc = pc
        self.ps = ps
        self.pca_con = sgg.SynGraspGen()
        self.last_joints = None
        self.debug_pub = rospy.Publisher('env_debug', PoseStamped, queue_size = 1)
        self.time = datetime.now().strftime("%d_%m_%Y_%H_%M")
        self.log_file = open(rospkg.RosPack().get_path('bimanual_handover') + "/logs/log" + self.time + ".txt", 'w')
        self.joint_order = self.fingers.get_active_joints()
        self.reset_timer = 0

    def step(self, action):
        reward = 0
        terminated = False
        scaled_action = np.array([action[0] * 1.5, action[1] * 0.65, action[2] * 1.25, action[3] * 0.95, action[4] * 0.45])
        try:
            result = self.pca_con.move_joint_config(scaled_action)
        except MoveItCommanderException as e:
            print(e)
            reward = -0.2
        if not result is MoveItErrorCodes.SUCCESS:
            reward = -0.2
        contact_points = self.__contact_points()
        current_joints = np.array(self.fingers.get_current_joint_values())
        normalized_joints = self.__normalize_joints(current_joints)
        observation = {}
#        observation["last_action"] = action
        #observation["finger_joints"] = normalized_joints
        observation["finger_contacts"] = contact_points
        ff = ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1']
        mf = ['rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1']
        rf = ['rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1']
        lf = ['rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1']
        th = ['rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1']
        finger_list = [ff, mf, rf, lf, th]
        if reward >= 0:
            joint_diff = 0
            for c in range(len(contact_points)):
                if contact_points[c] == 0:
                    finger = finger_list[c]
                    new_joints_finger = [normalized_joints[self.joint_order.index(joint)] for joint in finger]
                    last_joints_finger = [self.last_joints[self.joint_order.index(joint)] for joint in finger]
                    for i in range(len(last_joints_finger)):
                        if new_joints_finger[i] >= last_joints_finger[i]:
                            joint_diff += math.dist([new_joints_finger[i]], [last_joints_finger[i]])
                        else:
                            joint_diff += - math.dist([new_joints_finger[i]], [last_joints_finger[i]])
            '''
            for i in range(len(self.last_joints)):
                if new_joints[i] >= self.last_joints[i]:
                    joint_diff += math.dist([new_joints[i]], [self.last_joints[i]])
                else:
                    joint_diff += - math.dist([new_joints[i]], [self.last_joints[i]])
            '''
            reward = 0.03 * joint_diff
            reward += np.sum(contact_points) * 0.2
            if np.sum(contact_points) == 5:
                #reward = 1
                reward += 0.2
#                terminated = True
        self.last_joints = normalized_joints
        truncated = False
        info = {}
        print(scaled_action)
        print(reward)
        self.log_file.write("{}, {} \n".format(action, reward))
        self.reset_timer += 1
        if self.reset_timer == 100:
            terminated = True
        return observation, reward, terminated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.log_file.close()
        self.log_file = open(rospkg.RosPack().get_path('bimanual_handover') + "/logs/log"+ self.time + ".txt", 'a')
        self.reset_timer = 0
        self.fingers.set_named_target('open')
        self.fingers.go()
        self.pca_con.set_initial_hand_joints()
        contact_points = self.__contact_points()
        observation = {}
#        observation["last_action"] = np.array([0, 0, 0], dtype = np.float64)
        current_joints = np.array(self.fingers.get_current_joint_values())
        normalized_joints = self.__normalize_joints(current_joints)
        #observation["finger_joints"] = normalized_joints
        observation["finger_contacts"] = contact_points
        self.last_joints = normalized_joints
        info = {}
        return observation#, info

    def render(self):
        return

    def close(self):
        self.log_file.close()
        return

    def __normalize_joints(self, joints):
        '''
        J2: 0-pi/2
        J1: 0-pi/2
        J3: -15*pi/180 - pi/2
        J4: -20/180*pi - 20/180*pi
        LFJ5: 0 - 45/180*pi
        THJ1: -15/180*pi - pi/2
        THJ2: -40/180*pi - 40/180*pi
        THJ3: -12/180*pi - 12/180 * pi
        THJ4: 0 - 70/180*pi
        THJ5: -60/180pi - 60/180*pi
        '''
        ranges = {'rh_FFJ1': [0, math.pi/2], 'rh_FFJ2': [0, math.pi/2], 'rh_FFJ3': [-15/180*math.pi, math.pi/2], 'rh_FFJ4': [-20/180 * math.pi, 20/180 * math.pi], 'rh_MFJ1': [0, math.pi/2], 'rh_MFJ2': [0, math.pi/2], 'rh_MFJ3': [-15/180*math.pi, math.pi/2], 'rh_MFJ4': [-20/180 * math.pi, 20/180 * math.pi], 'rh_RFJ1': [0, math.pi/2], 'rh_RFJ2': [0, math.pi/2], 'rh_RFJ3': [-15/180*math.pi, math.pi/2], 'rh_RFJ4': [-20/180 * math.pi, 20/180 * math.pi], 'rh_LFJ1': [0, math.pi/2], 'rh_LFJ2': [0, math.pi/2], 'rh_LFJ3': [-15/180*math.pi, math.pi/2], 'rh_LFJ4': [-20/180 * math.pi, 20/180 * math.pi], 'rh_LFJ5': [0, 45/180 * math.pi], 'rh_THJ1': [-15/180*math.pi, math.pi/2], 'rh_THJ2': [-40/180*math.pi, 40/180*math.pi], 'rh_THJ3': [-12/180*math.pi, 12/180*math.pi], 'rh_THJ4': [0, 80/180*math.pi], 'rh_THJ5': [-60/180*math.pi, 60/180*math.pi]}
        normalized_joints = []
        for i in range(len(self.joint_order)):
            limits = ranges[self.joint_order[i]]
            shift = (limits[1] + limits[0])/2
            factor = limits[1] - shift
            normalized_joints.append((joints[i]-shift)/factor)
        normalized_joints = np.array(normalized_joints)
        return normalized_joints

    def __contact_points(self):
        cylinder = self.ps.get_objects(['can'])['can']
        h = cylinder.primitives[0].dimensions[0]
        r = cylinder.primitives[0].dimensions[1]
        pose = cylinder.pose
        p1 = deepcopy(pose.position)
        p1.z += h/2
        p1 = np.array([p1.x, p1.y, p1.z])
        p2 = deepcopy(pose.position)
        p2.z += -h/2
        p2 = np.array([p2.x, p2.y, p2.z])

        debug_pose = PoseStamped()
        debug_pose.header.frame_id = 'base_footprint'
        debug_pose.pose.position.x = p2[0]
        debug_pose.pose.position.y = p2[1]
        debug_pose.pose.position.z = p2[2]
        self.debug_pub.publish(debug_pose)

        tip_links = ['rh_ff_biotac_link', 'rh_mf_biotac_link', 'rh_rf_biotac_link', 'rh_lf_biotac_link', 'rh_th_biotac_link']
        tip_points = [[self.fingers.get_current_pose(link).pose.position.x, self.fingers.get_current_pose(link).pose.position.y, self.fingers.get_current_pose(link).pose.position.z] for link in tip_links]
        contact_points = [0, 0, 0, 0, 0]
        for x in range(len(tip_points)):
            if np.linalg.norm(np.cross(tip_points[x] - p1, tip_points[x] - p2))/np.linalg.norm(p2 - p1) <= r:
                contact_points[x] = 1
        contact_points = np.array(contact_points, dtype = np.int8)
        print(contact_points)
        return contact_points

