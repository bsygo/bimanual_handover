#!/usr/bin/env python3

import rospy
from stable_baselines3 import PPO
import rospkg
import bimanual_handover.syn_grasp_gen as sgg
import bimanual_handover.robot_setup_mover as setup_mover
from moveit_commander import roscpp_initialize, roscpp_shutdown
import numpy as np

def end_moveit():
    roscpp_shutdown()

def main():
    rospy.init_node("test_robot")
    roscpp_initialize("")
    rospy.on_shutdown(end_moveit)
    model_path = rospkg.RosPack().get_path('bimanual_handover') + "/models/ppo_model_10_05_2023_12_00"
    pub = rospy.Publisher("change_publish", bool)
    pub.publish(True)
    mover = setup_mover.RobotSetupMover()
    mover.move_fixed_pose_pc()
    # obs = tactile_feedback from env
    obs = np.array([0, 0, 0, 0, 0])
    model = PPO.load(model_path)
    action, _states = model.predict(obs)
    grasp_gen = sgg.SynGraspGen()
    grasp_gen.move_joint_config(action)
    pub.publish(False)
    mover.reset_fingers()
    mover.reset_arm()
    '''
    obs, reward, terminated, info = env.step(action)
    print(reward)
    '''

if __name__ == "__main__":
    main()
