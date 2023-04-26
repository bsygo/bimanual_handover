#!/usr/bin/env python3

import gym
import bimanual_handover.pc_env as pc_env
import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, PlanningSceneInterface
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sensor_msgs.msg import PointCloud2
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback

class ProgressBarCallback(BaseCallback):

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

class ProgressBarManager(object):

    def __init__(self, total_timesteps):
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):
        self.pbar = tqdm(total = self.total_timesteps)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

def end_moveit():
    roscpp_shutdown()

def main():
    rospy.init_node('ppo_trainer')
    roscpp_initialize('')
    rospy.on_shutdown(end_moveit)
    fingers = MoveGroupCommander('right_fingers')
    ps = PlanningSceneInterface()
    fingers.set_max_velocity_scaling_factor(1.0)
    fingers.set_max_acceleration_scaling_factor(1.0)
    rospy.loginfo('Waiting for pc.')
    pc = rospy.wait_for_message('pc/pc_filtered', PointCloud2, 20)
    rospy.loginfo('Setting up env.')
    env = pc_env.SimpleEnv(fingers, pc, ps) 
#    check_env(env)
    rospy.loginfo('Env check completed.')
    model = PPO("MultiInputPolicy", env, verbose = 1)
    rospy.loginfo('Starting learning.')
    with ProgressBarManager(2000) as callback:
        model.learn(total_timesteps=2000, callback=callback)
    model.save("test_ppo_model")
    obs = env.reset()
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    roscpp_shutdown()

if __name__ == "__main__":
    main()
