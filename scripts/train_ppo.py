#!/usr/bin/env python3

import gym
import bimanual_handover.pc_env as pc_env
import rospy
import rospkg
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, PlanningSceneInterface
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from sensor_msgs.msg import PointCloud2
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback

def end_moveit():
    roscpp_shutdown()

def main():
    test = False
    rospy.init_node('ppo_trainer')
    roscpp_initialize('')
    rospy.on_shutdown(end_moveit)
    rospack = rospkg.RosPack()
    path = rospack.get_path('bimanual_handover')
    fingers = MoveGroupCommander('right_fingers')
    ps = PlanningSceneInterface()
    fingers.set_max_velocity_scaling_factor(1.0)
    fingers.set_max_acceleration_scaling_factor(1.0)
    rospy.loginfo('Waiting for pc.')
    pc = rospy.wait_for_message('pc/pc_filtered', PointCloud2, 20)
    rospy.loginfo('Setting up env.')
    env = pc_env.SimpleEnv(fingers, pc, ps) 
    if not test:
    #    check_env(env)
        rospy.loginfo('Env check completed.')
        model = PPO("MultiInputPolicy", env, n_steps = 20, batch_size = 5, n_epochs = 100, verbose = 1, tensorboard_log=path + "/logs/tensorboard")
        rospy.loginfo('Starting learning.')
        model.learn(total_timesteps=2000, progress_bar = True)
        model.save(path + "/models/ppo_model")
    else:
        model = PPO.load("../ppo_model")
    obs = env.reset()
    action, _states = model.predict(obs)
    obs, reward, terminated, info = env.step(action)
    print(reward)
    roscpp_shutdown()

if __name__ == "__main__":
    main()
