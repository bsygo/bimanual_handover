#!/usr/bin/env python3

import gym
import bimanual_handover.pc_env as pc_env
import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sensor_msgs.msg import PointCloud2

def main():
    rospy.init_node('ppo_trainer')
    roscpp_initialize('')
    fingers = MoveGroupCommander('right_fingers')
    rospy.loginfo('Waiting for pc.')
    pc = rospy.wait_for_message('pc/pc_filtered', PointCloud2, 20)
    rospy.loginfo('Setting up env.')
    env = pc_env.SimpleEnv(fingers, pc) 
#    check_env(env)
    rospy.loginfo('Env check completed.')
    model = PPO("MultiInputPolicy", env, verbose = 1)
    rospy.loginfo('Starting learning.')
    model.learn(total_timesteps=25000)
    model.save("test_ppo_model")
    obs = env.reset()
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    roscpp_shutdown()

if __name__ == "__main__":
    main()
