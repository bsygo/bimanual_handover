#!/usr/bin/env python3

import gym
from datetime import datetime
import bimanual_handover.pc_env as pc_env
import rospy
import rospkg
import sys
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, PlanningSceneInterface
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from sensor_msgs.msg import PointCloud2
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import CheckpointCallback
import re


def end_moveit():
    roscpp_shutdown()

def main(argv):
    rospy.init_node('ppo_trainer')
    roscpp_initialize('')
    rospy.on_shutdown(end_moveit)
    rospack = rospkg.RosPack()
    path = rospack.get_path('bimanual_handover')
    fingers = MoveGroupCommander('right_fingers')
    ps = PlanningSceneInterface()
    fingers.set_max_velocity_scaling_factor(1.0)
    fingers.set_max_acceleration_scaling_factor(1.0)

    args_pattern = re.compile(r"(((?:--check)\s(?P<CHECK>True|False))?\s?((?:--test)\s(?P<TEST>True|False))?\s?((?:--model)\s(?P<MODEL>.*))?)")
    args = {}
    if len(argv) > 1:
        argv = " ".join(argv[1:])
    else:
        argv = ""
    if match_object := args_pattern.match(argv):
        args = {k: v for k, v in match_object.groupdict().items() if v is not None}
    if 'CHECK' in args:
        check = args ['CHECK']
    else:
        check = False
        rospy.loginfo('No check parameter specified, defaulting to False.')
    if 'TEST' in args:
        test = args['TEST']
    else:
        test = False
        rospy.loginfo('No test parameter specified, defaulting to False.')
    if 'MODEL' in args:
        model_path = args['MODEL']
    else:
        model_path = path + "/models/ppo_model"
        rospy.loginfo('No model parameter specified, defaulting to standard model.')

    rospy.loginfo('Waiting for pc.')
    pc = rospy.wait_for_message('pc/pc_filtered', PointCloud2, 20)
    rospy.loginfo('Setting up env.')
    env = pc_env.SimpleEnv(fingers, pc, ps)

    if not test:
        if check:
            check_env(env)
            rospy.loginfo('Env check completed.')
        date = datetime.now()
        str_date = date.strftime("%d_%m_%Y_%H_%M")
        checkpoint_callback = CheckpointCallback(save_freq = 500, save_path = path + "/models/", name_prefix = "ppo_checkpoint_" + str_date, save_replay_buffer = True, save_vecnormalize = True)
        model = PPO("MultiInputPolicy", env, n_steps = 20, batch_size = 5, n_epochs = 100, verbose = 1, tensorboard_log=path + "/logs/tensorboard")
        rospy.loginfo('Start learning.')
        model.learn(total_timesteps=2000, progress_bar = True, callback = checkpoint_callback)
        rospy.loginfo('Learning complete.')
        model.save(path + "/models/ppo_model_" + str_date)
    else:
        model = PPO.load(model_path)

    rospy.loginfo('Execute sample motion.')
    obs = env.reset()
    action, _states = model.predict(obs)
    obs, reward, terminated, info = env.step(action)
    print(reward)
    roscpp_shutdown()

if __name__ == "__main__":
    argv = rospy.myargv(argv = sys.argv)
    main(argv)
